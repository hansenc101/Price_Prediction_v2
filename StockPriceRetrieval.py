# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 10:14:31 2023

@author: hanse
"""

import yfinance as yf
import pandas as pd
import numpy as np

def get_price(tick,start='2021-10-01',end=None):
    return yf.Ticker(tick).history(start=start,end=end)['Close']

def get_prices(tickers,start='2021-10-01',end=None):
    df=pd.DataFrame()
    for s in tickers:
        df[s]=get_price(s,start,end)
    return df

def get_days_of_data(predict_date, feature_stocks, days=30): # NEEDS FIXED 
    end_date = pd.to_datetime(predict_date) # note, the end date is not included in the price range
    start_date = (end_date - pd.DateOffset(days=days))
    input_data = get_prices(tickers=feature_stocks, start=start_date.strftime('%Y-%m-%d')
                                , end=end_date.strftime('%Y-%m-%d'))
    while len(input_data) < days:
        end_date = start_date
        start_date = end_date - pd.DateOffset(days=days-len(input_data))
        temp = get_prices(tickers=feature_stocks, start=start_date.strftime('%Y-%m-%d')
                                    , end=end_date.strftime('%Y-%m-%d'))
        input_data = np.concatenate((temp, input_data))
    
    #Y = input_data[:,0]
    #y_min = np.min(Y)
    #y_max = np.max(Y)
    
    return input_data
