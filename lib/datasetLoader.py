#!/usr/bin/env python
# coding: utf-8

# In[14]:


from __future__ import division
import numpy as np
import torch
import os
import logging
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset, Sampler
from sklearn.preprocessing import StandardScaler
from lib.modules import time_features
from torch.utils.data.sampler import RandomSampler


logger = logging.getLogger('Dataset load')

class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='brw.csv',
                 target='WIND SPEED', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 168
            self.label_len = 0
            self.pred_len = 24
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(columns = ['date'], axis = 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]

        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        # print(seq_x, )
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)



class Dataset_all(object):
    def __init__(self, configs):
        self.configs = configs
        size = [configs.seq_len, configs.label_len, configs.pred_len]
        self.train_set = Dataset_Custom(configs.root_path, flag='train', size=size,
                        features=configs.features, data_path=configs.data_path,
                        target=configs.target, scale=configs.scale, timeenc=configs.timeenc, freq=configs.freq)
        
        self.val_set = Dataset_Custom(configs.root_path, flag='val', size=size,
                        features=configs.features, data_path=configs.data_path,
                        target=configs.target, scale=configs.scale, timeenc=configs.timeenc, freq=configs.freq)
        
        self.test_set = Dataset_Custom(configs.root_path, flag='test', size=size,
                        features=configs.features, data_path=configs.data_path,
                        target=configs.target, scale=configs.scale, timeenc=configs.timeenc, freq=configs.freq)

        self.train_loader = DataLoader(self.train_set, batch_size=configs.batch_size, sampler=RandomSampler(self.train_set), num_workers = configs.num_workers)
        self.validation_loader = DataLoader(self.val_set, batch_size=512, shuffle=False, num_workers = configs.num_workers)
        self.test_loader = DataLoader(self.test_set, batch_size=512, shuffle=False, num_workers = configs.num_workers)

    def _get_data(self, flag = 'train'):
        if flag == 'train' :
            return self.train_set, self.train_loader
        elif flag == 'val' :
            return self.val_set, self.validation_loader
        elif flag == 'test' :
            return self.test_set, self.test_loader
        else :
            logger.info(f'- Error Dataset Flag -')