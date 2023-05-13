#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 21:24:09 2023

@author: chris
"""

import torch
import torch.nn as nn
import LightningLearning as LL
import TorchUtils as tu
import numpy as np
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import TQDMProgressBar as bar
import StockPriceRetrieval as spr
from datetime import datetime

#%% Retrieve Stock Data 
predict_stock='msft'
feature_stocks=[predict_stock, 'tsla','meta','sony','amzn','nflx','gbtc','gdx','intc','dal',
                'c','goog','aapl','msft','ibm','hp','orcl','sap','crm','hubs','twlo']

#%% Load the trained model
model = torch.load('Trained_Model.pt')
model.eval()
preds = []

#%% Gather Input Data
days = 30 # The model was trained using 30 days of previous data 
date_format = "%Y-%m-%d"
should_predict = True
while should_predict:
    while True:
        date = input("What date would you like to predict? (YYYY-MM-DD): ")
        try:
            # Parse the input date using datetime.strptime
            datetime.strptime(date, date_format)
            break  # Break out of the loop if the input is in the correct format
        except ValueError:
            print("Invalid date format. Please enter a date in the format YYYY-MM-DD.")
    print(f"Gathering historical stock data from the past {days} days from {date}...\n")
    input_data = spr.get_days_of_data(predict_date=date, feature_stocks=feature_stocks, days=days)
    y_min, y_max = tu.get_training_scale()
    print('Done!')
    
    # Query the model for a prediction
    x = torch.tensor(input_data, dtype=torch.float).unsqueeze(1) #LSTM expects input of size (sequence_length, batch_size, n_features/sample)

    # Determine if using GPU for neural network computations
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
        x = x.to(device)
        model = model.to(device)
        
    prediction = model.forward(x).cpu().detach().numpy()
    prediction = tu.min_max_undo(prediction, y_min, y_max)
    preds.append(prediction)
    print(f'The predicted closing price of {predict_stock.upper()} for {date} is: {prediction}')
    
    response = input('Do you want to perform another prediction? (y/n)')
    if (response.lower()=='n') or (response.lower()=='no'): 
        should_predict = False
    
input('Press [Enter] to quit...')
#%% Visualize Predictions




