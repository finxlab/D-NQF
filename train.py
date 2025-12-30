#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:
import time
import os

import argparse
import logging
import os

import numpy as np
from numpy.linalg import inv
from scipy.stats import chi2
import torch
from torch import nn
import torch.optim as optim
from tqdm import tqdm

# import lib
from lib.datasetLoader import *
from lib.Model import *
from lib.modules import *
from lib.Metric import *


logging.basicConfig(
    format='%(asctime)s %(levelname)s:%(message)s',
    level=logging.DEBUG,
    datefmt='%m/%d/%Y %I:%M:%S %p',
)
logger = logging.getLogger('Timemixer.Train')

LOSS_DICT = {
    "ND": ND,
    "NRMSE": NRMSE,
    "CRPS": CRPS,
    "QLOSS": QLOSS,
    "ECRPS": ECRPS,
    "QCL": QCL,
    "MSE": MSE,
    "MAE": MAE,
    "MSPE": MSPE,
    "MAPE": MAPE,
    "Rsquare": Rsquare
    }



    
def test_predict(model, test_set, test_loader, configs):
    all_outputs = []  # to collect outputs
    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm(test_loader)):

            batch_x = batch_x.float().to(configs.device)
            batch_y = batch_y.float().to(configs.device) # N, T, 1
            
            batch_x_mark = batch_x_mark.float().to(configs.device)
            batch_y_mark = batch_y_mark.float().to(configs.device)
            N, T, _ = batch_y.shape

            
            alpha = torch.concat([torch.arange(1/configs.n_quantiles, 0.999, 1 / configs.n_quantiles), torch.Tensor([0.999])]).to(configs.device)[None, None, :, None]
            alpha = alpha.repeat(N,T,1,1)
            outputs = model(batch_x, batch_x_mark, batch_y_mark, alpha[:, 0])

            outputs = outputs.detach().cpu().numpy()
            batch_y = batch_y.detach().cpu().numpy()

            if configs.scale == True :
                outputs = test_set.inverse_transform(outputs.reshape(-1,1)).reshape(outputs.shape)
                batch_y = test_set.inverse_transform(batch_y.reshape(-1,1)).reshape(batch_y.shape)
            all_outputs.append(outputs)

        # After the loop, concatenate all batch outputs into one array
        all_outputs = np.concatenate(all_outputs, axis=0)

    return all_outputs
def evaluate(model, test_set, test_loader, configs, loss_type = ['ND']):
    
    model.eval()
    with torch.no_grad():
        summary_metric = {}
        eval_batch = {}
        
        for loss in loss_type :
            if loss == 'NRMSE' :
                eval_batch[loss] = np.zeros(3)
            else :
                eval_batch[loss] = np.zeros(2)

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm(test_loader)):

            batch_x = batch_x.float().to(configs.device)
            batch_y = batch_y.float().to(configs.device) # N, T, 1
            
            batch_x_mark = batch_x_mark.float().to(configs.device)
            batch_y_mark = batch_y_mark.float().to(configs.device)
            N, T, _ = batch_y.shape

            
            alpha = torch.concat([torch.arange(1/configs.n_quantiles, 0.999, 1 / configs.n_quantiles), torch.Tensor([0.999])]).to(configs.device)[None, None, :, None]
            alpha = alpha.repeat(N,T,1,1)
            # alpha -= 1/2
            
            # ND, RMSE evaluation
            outputs = model(batch_x, batch_x_mark, batch_y_mark, alpha[:, 0])
            outputs = outputs[:, -configs.pred_len:,] # N, T, M

            outputs = outputs.detach().cpu().numpy()
            batch_y = batch_y.detach().cpu().numpy()

            if configs.scale == True :
                outputs = test_set.inverse_transform(outputs.reshape(-1,1)).reshape(outputs.shape)
                batch_y = test_set.inverse_transform(batch_y.reshape(-1,1)).reshape(batch_y.shape)
            for loss in loss_type :
                if loss == 'NRMSE' :
                    diff, summation, index_n = LOSS_DICT[loss](outputs, batch_y)
                    eval_batch[loss][0] += diff
                    eval_batch[loss][1] += summation
                    eval_batch[loss][2] += index_n
                                    
                else :
                    upper, lower = LOSS_DICT[loss](outputs, batch_y)
                    eval_batch[loss][0] += upper
                    eval_batch[loss][1] += lower
                    
        print(outputs[0,0], batch_y[0,0], outputs[0,0, 100])

        for loss in loss_type :
            if loss == 'NRMSE' :
                summary_metric[loss] = ((eval_batch[loss][0] / eval_batch[loss][2])**(1/2)) / (eval_batch[loss][1] / eval_batch[loss][2])
            else :
                summary_metric[loss] = eval_batch[loss][0]/ eval_batch[loss][1] 
                
    return summary_metric




def modelTrain(model: nn.Module,
          optimizer: optim,
          train_loader: DataLoader,
          configs)  :

    model.train()
    loss_epoch = np.zeros(len(train_loader))
    criterion = CRPS_loss(configs.n_quantiles,configs.device) # random shuffle alpha

    # criterion2 = nn.MSELoss()
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()

        batch_x = batch_x.float().to(configs.device)
        batch_y = batch_y.float().to(configs.device)
        batch_x_mark = batch_x_mark.float().to(configs.device)
        batch_y_mark = batch_y_mark.float().to(configs.device)
        N, T, _ = batch_y.shape

        alpha = torch.rand(N,T,configs.n_quantiles).to(configs.device)
        alpha_sorted, indices = torch.sort(alpha, dim = -1)
        # print(alpha_sorted)
        alpha = torch.concat([torch.arange(1/configs.n_quantiles, 0.999, 1 / configs.n_quantiles), torch.Tensor([0.999])]).to(configs.device)[None, :, None] # N M 1
        alpha = alpha.repeat(N,1,1)
        # alpha -= 1/2
        outputs = model(batch_x, batch_x_mark, batch_y_mark, alpha)
        # print(outputs.shape, batch_y.shape)
        # print(outputs.shape, batch_y.shape, alpha.shape)
        loss = criterion(outputs, batch_y, alpha.permute(0,2,1))
        # print(outputs.shape, batch_y.shape)
        # loss += criterion2(outputs[:, :, 100:101], batch_y)
        if i % 300 == 0 :
            print(loss.item())
        loss.backward()
        optimizer.step()
        loss_epoch[i] = loss.item()

    return loss_epoch


def train_and_evaluate(model: nn.Module,
                       dataset_all: object,
                       optimizer: optim, 
                       configs) :

    logger.info('begin training and evaluation')
    
    # dataset
    train_set, train_loader = dataset_all._get_data('train')
    val_set, validation_loader = dataset_all._get_data('val')
    test_set, test_loader = dataset_all._get_data('test')


    path = os.path.join(configs.model_dir, configs.setting)

    if not os.path.exists(path):
        os.makedirs(path)

    train_len = len(train_loader)
    crps_summary = np.zeros(configs.train_epochs)

    loss = configs.loss
    evaluation_summary = {}
    loss_summary = np.zeros((train_len * configs.train_epochs))
    early_stopping = EarlyStopping(patience=configs.patience, verbose=True)
    
    for epoch in tqdm(range(configs.train_epochs)):
        logger.info('Epoch {}/{}'.format(epoch + 1, configs.train_epochs))
        # Model Train
        loss_summary[epoch * train_len:(epoch + 1) * train_len] = modelTrain(model, optimizer, train_loader, configs)


        # Evaluate Model
        summary_val = evaluate(model, val_set, validation_loader, configs, loss_type =["ND", "NRMSE", "Rsquare", "CRPS", "QLOSS", "ECRPS", "QCL", "MSE", "MAE", "MSPE", "MAPE"])

        evaluation_summary[epoch] = summary_val


        val_loss = summary_val[loss]
        early_stopping(val_loss, model, path)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    logger.info(f'Current Best Loss is:  {early_stopping.best_score}')

    evaluation_summary['best_score'] = early_stopping.best_score
    json_path =  path + '/' + 'validation_metric.json'
    save_dict_to_json(evaluation_summary, json_path)

    configs_dict = vars(configs)
    configs_dict['device'] = str(configs_dict['device'])
    configs_dict['best score'] = early_stopping.best_score
    json_path =  path + '/' + 'configs.json'
    save_dict_to_json(configs_dict, json_path)

