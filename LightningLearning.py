# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 12:11:48 2023

@author: hanse
"""

"""
1a. Create a subclass of pytorch_lightning.LightningModule. It should include 
__init__, training_step, validation_step, configure_optimizers in the class. 
(6 points)

"""
import torch
from torch import optim, nn, stack
import pytorch_lightning as pl
import LSTM_Net as lstm
import numpy as np
#import TorchUtils as tu

class Config(): # I should change this to a dictionary instead of a class
    def __init__(self, n_input_size=1, hidden_size=1, n_layers=1, lr=1e-4, dropout=0, y_min=0, y_max=1):
        self.learning_rate = lr
        self.dropout = dropout
        self.n_input_size = n_input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

default = Config()

# Computational Code goes into LightningModule
class StockLightningModule(pl.LightningModule):
    def __init__(self, config): # model architecture into __init()__
        super().__init__()
        # our untrained neural network
        self.net = lstm.PriceNet(config=config)
        self.lr = config.learning_rate # wandb tuning for learning rate
        self.val_loss_data = []
        self.train_loss_data = []
        self.val_target_data = np.empty(0)
        self.val_pred = np.empty(0)
        self.used_gpu = False
        if torch.cuda.is_available():
            self.used_gpu = True
    
    # Set forward hook; in lightning, forward() defines the prediction and interference actions
    def forward(self, x):
        print('\n Input x: ', x, '\n')
        embedding = self.net(x)
        return embedding
    
    # Training logic goes into training_step LightningModule hook
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch 
        x=x.permute(2, 0, 1)# reshape while retaining data integrity 
        # LSTM expects input of size (sequence_length, batch_size, n_features/sample)
        
        output = self.net(x)
        loss = nn.functional.mse_loss(output,y)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        self.train_loss_data.append(loss)
        return loss
    
    # Validation logic goes into validation_step LightningModule hook
    def validation_step(self, val_batch, batch_idx):
        x, y  = val_batch 
        x=x.permute(2, 0, 1) # reshape while retaining data integrity 
        # LSTM expects input of size (sequence_length, batch_size, n_features/sample)
        output = self.net(x)
        loss = nn.functional.mse_loss(output,y) # compute loss
        # calling this from validation_step will automatically accumulate
        # and log at the end of the epoch
        self.log('val_loss', loss) 
        self.val_loss_data.append(loss)
        
        # Collect target and prediction data
        Y = y.view(-1).squeeze()
        Output = output.view(-1).squeeze()
        if self.used_gpu:
            Y = np.array([tensor.cpu().detach().numpy() for tensor in y])
            Output = np.array([tensor.cpu().detach().numpy() for tensor in output])
        else:
            Y = np.array([tensor.detach().numpy() for tensor in y])
            Output = np.array([tensor.detach().numpy() for tensor in output])
            
        self.val_target_data = np.concatenate((self.val_target_data,Y), axis=0)
        self.val_pred = np.concatenate((self.val_pred,Output), axis=0)
    
    # Optimizers go into configure_optimizers LightningModule hook
    # self.parameters will contain parameters from neural net
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    # Retrieve Loss and Output Data
    def get_train_loss_data(self):
        return stack(self.train_loss_data)
    
    def get_val_loss_data(self):
        return stack(self.val_loss_data)
    
    def get_val_predictions(self): # NEEDS FIXED
        return self.val_pred
    
    def get_val_targets(self): # NEEDS FIXED
        return self.val_target_data
    
    def save(self, file_name='Trained_Model.pt'):
        torch.save(self.net, file_name)