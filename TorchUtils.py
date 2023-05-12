# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 10:23:13 2023

@author: hanse
"""

"""
1b. Create a subclass of pytorch_lightning.LightningDataModule. It should 
include __init__, train_dataloader, and val_dataloader in the class. (4 points)

"""

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader,Dataset
import torch.utils.data as data
import numpy as np


def min_max_scale(x, axis=1):
    x_max = np.max(x, axis=axis, keepdims=True)
    x_min = np.min(x, axis=axis, keepdims=True)
    x_scaled = (x - x_min)/(x_max - x_min)
    return x_scaled, x_min, x_max

def min_max_undo(x_scaled, min, max):
    x_origin = (x_scaled*(max - min)) + min
    return x_origin

def np_get_tensor_list(tensor_list):
    data = np.empty(0)
    if torch.cuda.is_available():
        # get loss data off of gpu to cpu and convert to np.arrays
        for tensor in tensor_list:
            temp = np.array(tensor.cpu().detach().numpy(), dtype=float)
            data = np.concatenate((data, temp))
    else: 
        # Take loss data and convert to np.arrays
        for tensor in tensor_list:
            temp = np.array(tensor.detach().numpy(), dtype=float)
            data = np.concatenate((data, temp))
    return data

class StockDataset(Dataset):
    def __init__(self,X,Y,days): # X and Y must be np.array() types
        self.X = X # 100
        self.Y = Y.reshape(-1) # 1
        self.days = days # days ahead for prediction
        
    def __len__(self):
        return (len(self.Y)-self.days)
        
    def __getitem__(self,index):
        x=self.X[:,index:index+self.days]
        y=self.Y[index+self.days]
        return x,y

class StockDataModule(pl.LightningDataModule):
    def __init__(self, stockData, batch_size = 0):
        self.num_workers = 16
        train_set_size = int(len(stockData)*0.7)
        valid_set_size = int(len(stockData)*0.15)
        test_set_size = len(stockData)-train_set_size-valid_set_size

        self.train_set, self.valid_set, self.test_set = data.random_split(stockData,[train_set_size,valid_set_size,test_set_size],\
                                                      generator=torch.Generator().manual_seed(42))  
        if batch_size==0:
            self.batch_size = test_set_size # use entire dataset as batch
        else:
            self.batch_size = batch_size
    
    def train_dataloader(self):
        train_dataloader = DataLoader(self.train_set,batch_size=self.batch_size,shuffle=True, num_workers=self.num_workers)  # input:(n_predict_stocks,days), label:1
        return train_dataloader
    
    def val_dataloader(self):
        val_dataloader = DataLoader(self.valid_set,batch_size=self.batch_size,shuffle=False, num_workers=self.num_workers)  # input:(n_predict_stocks,days), label:1
        return val_dataloader
    
    def test_dataloader(self):
        test_dataloader = DataLoader(self.test_set,batch_size=self.batch_size,shuffle=False, num_workers=self.num_workers)  # input:(n_predict_stocks,days), label:1
        return test_dataloader