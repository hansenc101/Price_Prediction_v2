#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 12:39:45 2023

@author: chris

This file will simply save the stock prices in a csv and store them so that they 
can be retrieved later without internet access. 
"""

import StockPriceRetrieval as spr
import numpy as np


predict_stock='msft'
feature_stocks=[predict_stock, 'tsla','meta','sony','amzn','nflx','gbtc','gdx','intc','dal',
                'c','goog','aapl','msft','ibm','hp','orcl','sap','crm','hubs','twlo']
n_features = len(feature_stocks)-1

start = '2015-01-01'

# getting data
print('Retrieving stock data...')
start_date='2018-01-01'
allX=spr.get_prices(feature_stocks,start=start_date)
ally=spr.get_prices([predict_stock],start=start_date)
allX = allX.to_numpy().transpose().astype(np.float32)
ally = ally.to_numpy().astype(np.float32)
length = np.size(allX,1)
X = allX[0:length-1,:]
Y = ally[1:length]
Y_same_day = ally[0:length-1,:]


x_file_path = 'x_data.csv'
y_file_path = 'y_data_next_day.csv'
y_same_file_path = 'y_data_same_day.csv'

np.savetxt(x_file_path, X, delimiter=',')
np.savetxt(y_file_path, Y, delimiter=',')
np.savetxt(y_same_file_path, Y_same_day, delimiter=',')

print('Data was received and stored in working directory.')