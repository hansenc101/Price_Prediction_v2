# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 10:14:31 2023

@author: hanse
"""

import yfinance as yf
import pandas as pd

def get_price(tick,start='2021-10-01',end=None):
    return yf.Ticker(tick).history(start=start,end=end)['Close']

def get_prices(tickers,start='2021-10-01',end=None):
    df=pd.DataFrame()
    for s in tickers:
        df[s]=get_price(s,start,end)
    return df