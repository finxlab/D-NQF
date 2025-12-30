#!/usr/bin/env python
# coding: utf-8

# In[110]:


from typing import Optional, Tuple
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import logging
import json
import shutil
from lib.modules import *
logger = logging.getLogger('Benchmark.Model') 

class PositiveLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(PositiveLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.ones(out_features, in_features))
        self.bias= nn.Parameter(torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight)

    def forward(self, input):
        return nn.functional.linear(input, torch.square(self.weight), self.bias)



class NQF(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim = 32):
        super(NQF, self).__init__()
        NQF_layers = []
        NQF_layers.extend([
                PositiveLinear(input_dim, hidden_dim),
                nn.Sigmoid(),
                PositiveLinear(hidden_dim, output_dim),
            ])
        self.NQF_model = nn.Sequential(*NQF_layers)

    def forward(self, x, alpha):
        B,T,M = x.shape
        q = torch.logit(alpha)
        return self.NQF_model(torch.concat([x, q], axis = -1))

class DLinear(nn.Module):
    """DLinear

    Args:
        configs : c_out, seq_len, pred_len
    References:
        - [Zeng, Ailing, et al. "Are transformers effective for time series forecasting?." Proceedings of the AAAI conference on artificial intelligence. Vol. 37. No. 9. 2023."](https://ojs.aaai.org/index.php/AAAI/article/view/27695)
    """

    def __init__(
        self,
        configs
    ):
        super(DLinear, self).__init__()

        # Architecture
        self.moving_avg_window = configs.moving_avg

        self.c_out = 1
        self.n_quantiles = configs.n_quantiles
        self.enc_in = 1
        self.dec_in = 1
        self.input_size = configs.seq_len
        self.h = configs.pred_len
        # Decomposition
        self.decomp = SeriesDecomp(self.moving_avg_window)

        self.linear_trend = NQF(
            self.input_size+1, self.c_out * self.h
        )
        self.linear_season = NQF(
            self.input_size+1, self.c_out * self.h
        )
    def forward(self, x_enc, x_mark_enc, x_mark_dec, alpha):
        # Parse windows_batch
        insample_y = x_enc.squeeze(-1) # B T 
        
        # Parse inputs
        seasonal_init, trend_init = self.decomp(insample_y)
        trend_part = self.linear_trend(trend_init[:, None].repeat(1, self.n_quantiles, 1), alpha)
        seasonal_part = self.linear_season(seasonal_init[:, None].repeat(1, self.n_quantiles, 1), alpha)

        # Final
        forecast = trend_part + seasonal_part
        return forecast.permute(0,2,1)


def save_checkpoint(state, is_best, epoch, checkpoint, ins_name=-1):
    '''Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'
    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
        ins_name: (int) instance index
    '''
    if ins_name == -1:
        filepath = os.path.join(checkpoint, f'epoch_{epoch}.pth.tar')
    else:
        filepath = os.path.join(checkpoint, f'epoch_{epoch}_ins_{ins_name}.pth.tar')
    if not os.path.exists(checkpoint):
        logger.info(f'Checkpoint Directory does not exist! Making directory {checkpoint}')
        os.mkdir(checkpoint)
    
    # torch.save(state, filepath)
    # logger.info(f'Checkpoint saved to {filepath}')
    
    if is_best:
        torch.save(state, os.path.join(checkpoint, f'ins_{ins_name}_best.pth.tar'))
        logger.info('Best checkpoint copied to best.pth.tar')

def load_checkpoint(checkpoint, model, optimizer = None):
    '''Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
        optimizer_dain: (torch.optim) optional: resume optimizer from checkpoint
        gpu: which gpu to use
    '''
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"File doesn't exist {checkpoint}")
        
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint, map_location='cuda')
        
    else:
        checkpoint = torch.load(checkpoint, map_location='cpu')
        
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])
        optimizer.param_groups[0]['capturable'] = True

    return checkpoint

    
def save_dict_to_json(d, json_path):
    '''Saves dict of floats in json file
    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    '''
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        # d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def load_json(path) :
    with open(path, 'r') as f:
        json_data = json.load(f)
    return json_data


