# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 19:43:15 2021
Tesis Giampaolo

@author: gema_
"""

#Importing Libraries and csv
import pandas as pd
import numpy as np
from scipy.stats import norm
import time 
import matplotlib.pyplot as plt
ds = pd.read_csv('L3_options_20190815.csv')

#Creating the target variable ' Average Price' as the average of Bid and Ask price
ds['Average Price']=(ds['Bid']+ds['Ask'])/2
ds.drop('Bid', axis=1, inplace=True)
ds.drop('Ask', axis=1, inplace =True)

date_format = "%m/%d/%Y"
#Creating the variable 'Time T oM aturity' as a fraction of 252 yearly working days
ds['StartDay'] = pd.to_datetime(ds[' DataDate']).sub(pd.Timestamp('2019 01 01')).dt.days
ds['ExpirationDay'] = pd.to_datetime(ds['Expiration']).sub(pd.Timestamp('2019 01 01')).dt.days
ds['TimeToMaturity'] = (ds['ExpirationDay'] - ds['StartDay'])/252

ds.drop('StartDay', axis=1, inplace=True)
ds.drop('ExpirationDay', axis=1, inplace=True)

#Risk Free rate, the US 3 months treasury bill rate
rf = 0.0187
#Deleting the unused columns
ds.drop('Volume', axis=1, inplace=True)
ds.drop('OpenInterest', axis=1, inplace=True)
ds.drop('IVBid', axis=1, inplace=True)
ds.drop('IVAsk', axis=1, inplace=True)
ds.drop('UnderlyingSymbol', axis=1, inplace = True)
ds.drop('Expiration', axis=1, inplace=True)
ds.drop(' DataDate', axis=1, inplace=True)
ds.drop('Last', axis=1, inplace=True)
ds.drop('Delta', axis=1, inplace=True)
ds.drop('Gamma', axis=1, inplace=True)
ds.drop('Theta', axis=1, inplace=True)
ds.drop('Vega', axis=1, inplace=True)
ds.drop('OptionSymbol', axis=1, inplace=True)

#train test split
train_ds = ds.sample(frac=0.8,random_state=0)
test_ds = ds.drop(train_ds.index)



#Defining Black and Scholes formula
def BS_model (S, K, T, r, sigma, option = 'call'):
    d1 = (np.log( S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    if option == 'call':
        result = (S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    if option == 'put':
        result = (K * np.exp(-r * T) * norm.cdf(-d2)-S * norm.cdf(-d1))
    return result

#Applying Black S choles formula to test set and counting time to compute
start = time.time()
test_ds['BS_Price']= np.where(test_ds['Type']=='call',
						BS_model(test_ds['UnderlyingPrice'],test_ds['Strike'],test_ds['TimeToMaturity'],rf,test_ds['IVMean'], option = 'call'),
						BS_model (test_ds['UnderlyingPrice'],	test_ds['Strike'],test_ds['TimeToMaturity'],rf, test_ds['IVMean'], option = 'put'))
#Calculating the mean absolute error of the model
test_ds['Absolute_error']= np.absolute(test_ds['Average Price']- test_ds['BS_Price'])
end = time.time()
MAE = np.mean(test_ds['Absolute_error'])
test_ds['Index'] = test_ds.index
plt.scatter(test_ds['Index'], test_ds['Average Price'],label = "Real")
plt.scatter(test_ds['Index'], test_ds['BS_Price'],label = "Pred")
plt.legend()



