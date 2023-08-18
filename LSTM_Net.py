# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 21:08:53 2023

@author: Christopher Hansen
email: hansenc101@gmail.com
"""

import torch
import torch.nn as nn

""" ==== Class: PriceNet ====
    Inputs: 
        n_inputs: number of inputs into the LSTM model
        n_hidden_units: number of nodes for each hidden layer of LSTM model
        n_layers: number of layers of LSTM model
        config: data structure containing additional parameters that can be modified
          - dropout_rate: rate of dropout for the hidden layer nodes
          - learning_rate: learning rate of LSTM model
    Outputs:
        Creates an instance of the PriceNet class. This class contains
        the functions and internal objects needed to create a neural network. 
        This neural network is able to take a vector of N samples and predict the 
        next sample. This class was developed for single point stock prediction.
"""
class PriceNet(torch.nn.Module):
    def __init__(self, config):
        super(PriceNet, self).__init__()
        self.dropout = config.dropout
        self.learning_rate = config.learning_rate
        n_input_features = config.n_input_size[2]
        #batch_size = config.n_input_size[1]
        #n_samples = config.n_input_size[0]
        n_hidden_units = config.hidden_size
        n_layers = config.n_layers
        
        #LSTM Layer(s)
        self.lstm_net = nn.LSTM(input_size=n_input_features, hidden_size=n_hidden_units,
                                  num_layers=n_layers, dropout=self.dropout)
        
        # Output layer for scalar prediction 
        self.output_layer = nn.Linear(n_hidden_units, 1)  
        torch.nn.init.xavier_uniform_(self.output_layer.weight) # intialize the weights of output layer
    
    def forward(self, x):
        # Apply LSTM network
        output, (h_0, c_0) = self.lstm_net(x) 
        
        output = output[-1,:] # grab the last time step
        output = self.output_layer(output)  # Apply Linear Layer
        output = output.squeeze() # change from size (batch_size,1) to (batch_size)
        return output
