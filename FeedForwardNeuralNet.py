# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 17:35:52 2023

@author: Christopher Hansen
"""
import torch
import torch.nn.functional as F

""" ==== Class: NeuralNetwork ====
    Inputs: 
        n_feature: number of input features into the neural network
        n_hidden_nodes: number of nodes for the hidden layers (static)
        n_hidden_layers: number of hidden layers of the neural network
        n_output: number of output channels of the neural network
        dropout_rate: rate of dropout for the hidden layer nodes
    Outputs:
        Creates an instance of the NeuralNetwork class. This class contains
        the functions and internal objects needed to create a neural network. 
        The network uses ReLU activation functions and processes each node as 
        y = ax+b, where x is the input data, y is the output data, and b is a 
        learnable bias weight. When the class is initially created, the network
        is untrained. 
"""
class NeuralNetwork(torch.nn.Module):
    def __init__(self, n_feature, n_hidden_nodes, n_hidden_layers, n_output, dropout_rate=0):
        super(NeuralNetwork, self).__init__()
        
        # input layer
        self.input_layer = torch.nn.Linear(n_feature, n_hidden_nodes) # create the input layer
        torch.nn.init.xavier_uniform_(self.input_layer.weight) # initialize weights of input layer
        
        # create an empty, iterable list of linear layers to make up the hidden layers
        self.hidden_layers=torch.nn.ModuleList() 
        
        for i in range(n_hidden_layers): # iterate through and create all hidden layers
            layer = torch.nn.Linear(n_hidden_nodes, n_hidden_nodes)  # create a hidden linear layer
            torch.nn.init.xavier_uniform_(layer.weight) # initialize weights of the hidden layer
            
            # add initialized linear layer to the list of modules
            self.hidden_layers.add_module('hidden_layer_{}'.format(i), torch.nn.Sequential(layer,torch.nn.Dropout(p=dropout_rate)))
            
        # create the output layer    
        self.output_layer = torch.nn.Linear(n_hidden_nodes, n_output)
        torch.nn.init.xavier_uniform_(self.output_layer.weight) # intialize the weights of output layer
        
        self.dropout = torch.nn.Dropout(p=dropout_rate) # setting dropout rate
        
    def forward(self, x):
        x=x.view(x.size(0),-1)
        x = F.relu(self.input_layer(x)) # Input and process at input layer
        for layer in self.hidden_layers:
            x = F.relu(layer(x)) # Process the hidden layers
        x = F.relu(self.output_layer(x)) # Process the output layers
        return x