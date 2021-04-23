# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 00:39:02 2021

@author: gema_
"""

import pandas as pd
import numpy as np
from scipy.stats import norm
import torch 
import time
import matplotlib.pyplot as plt
import seaborn as sns

ds =  pd.read_csv('Deribit_w_IV.csv',dtype={"timestamp":"string","price":"string","instrument_name":"string","index_price":"string","direction":"string","amount":"string","time_trade":"string","strike":"string","type":"string","time_create":"string","time_expire":"string","date":"string","irate":"string","price_USD":"string","iv_Tbill":"string","tt":"string"})
dummy=pd.get_dummies(ds['direction'])
ds=pd.concat([ds,dummy], axis=1)
dummy=pd.get_dummies(ds['type'])
dummy.head()
ds=pd.concat([ds,dummy],axis=1)

ds.drop('timestamp', axis=1, inplace=True)
ds.drop('instrument_name', axis=1, inplace=True)
ds.drop('time_trade', axis=1, inplace=True)
ds.drop('iv_Tbill', axis=1, inplace=True)

ds['time_create']=ds['time_create'].str[:10] #creation date
ds['time_expire']=ds['time_expire'].str[:10] #expiration date


ds['time_create'] = pd.to_datetime(ds['time_create'])
ds['time_expire'] = pd.to_datetime(ds['time_expire'])
ds['time'] = ds['time_expire'] - ds['time_create']
ds['time'] = ds['time'].dt.days.astype('int16')
ds.to_csv(r'C:\Users\gema_\OneDrive\Escritorio\Tesis\BTC_OPTIONS_VERSION2.csv', index = False)

descriptive = ds.describe()
'''==================== Dataset completed========================='''

'''======
Black-Scholes Model
call = 1
put = 0
========'''

def BS_model (S, K, T, r, sigma, vanilla = 1):
    d1 = (np.log( S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log( S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    nd1=norm.cdf(d1)
    nd2=norm.cdf(d2)
    if vanilla == 1:
        price = (S * nd1 - K * np.exp(-r * T) * nd2)
    if vanilla == 0:
        price = (K * np.exp(-r * T) * norm.cdf(-d2)-S * norm.cdf(-d1))
    return price

print(BS_model(18,22,1,0.05,0.3))

'''======
Getting the prices of BS model 
========'''

def get_bsm_price(data,sigma):
    #create the list
    a=[]    
    #Get tensor size
    sizeT = list(data.size())[0]  
    for i in range(sizeT):
        S = data[i][1].item() #
        K = data[i][2].item() #
        T = data[i][4].item()/365 # days/
        r = data[i][3].item()
        call_price = BS_model(S,K,T,r,sigma)
        a.append(call_price)
    #Convert to array 
    arrayOut = np.array(a)
    return arrayOut

'''===Black Scholes Model Execution'''

ds=pd.read_csv(r'BTC_OPTIONS_VERSION2.csv')
ds.drop('direction', axis=1, inplace=True)
ds.drop('type', axis=1, inplace=True)
ds['time_create']=pd.to_datetime(ds['time_create'])
ds['time_expire']=pd.to_datetime(ds['time_expire'])

boolean_option = ds['call']==1
ds = ds[boolean_option]
boolean_option = ds['time']>0
ds = ds[boolean_option]

ds.dtypes

data = ds.iloc[:,[11,3,6,10,18]] #neccesary data

data_tensor = torch.tensor(data.values)

des_std=ds['index_price'].std()
sigma_value = 0.034 #https://www.buybitcoinworldwide.com/es/indice-de-volatilidad/

start = time.time()
bsm_prices=get_bsm_price(data_tensor,sigma=sigma_value)
end = time.time()

ds.reset_index(inplace=True)
numer_records=ds.index.to_list()
number_records= np.array(numer_records)
predictions= ds['price_USD']

ds['BSM_price']= bsm_prices
ds['Absolute_error']= np.absolute(ds['price_USD']- ds['BSM_price'])
MAE = np.mean(ds['Absolute_error'])
print(MAE)


#Plot of true vs predicted
plt.plot(number_records, predictions,label = "true")
plt.plot(number_records, bsm_prices, label="pred")
plt.legend()





















