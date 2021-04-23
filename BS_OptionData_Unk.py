# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 00:25:40 2021

@author: gema_
"""
import pandas as pd
import numpy as np
from scipy.stats import norm
import torch 
import time
import matplotlib.pyplot as plt

ds = pd.read_csv('btc_options_call.csv',dtype={"exchange":"string","symbol":"string","timestamp":"string","loca_timestamp":"string","type":"string","strike_price":"string","expiration":"string","open_interest":"string","last_price":"string","bid_price":"string","bid_amount":"string","bid_iv":"string","ask_price":"string","ask_amount":"string","ask_iv":"string","mark_price":"string","mark_iv":"string","underlying_index":"string","underlying_price":"string","delta":"string","gamma":"string","vega":"string","theta":"string","rho":"string"})

boolean_option = ds['symbol'].str.startswith('BTC')
btc_options = ds[boolean_option]

boolean_option = btc_options['type']=='call'
btc_options_call = btc_options[boolean_option]

btc_options_call = pd.read_csv('btc_options_call.csv',dtype={"exchange":"string","symbol":"string","timestamp":"string","loca_timestamp":"string","type":"string","strike_price":"string","expiration":"string","open_interest":"string","last_price":"string","bid_price":"string","bid_amount":"string","bid_iv":"string","ask_price":"string","ask_amount":"string","ask_iv":"string","mark_price":"string","mark_iv":"string","underlying_index":"string","underlying_price":"string","delta":"string","gamma":"string","vega":"string","theta":"string","rho":"string"})

proof=btc_options_call['expiration'].to_numpy(dtype="double")
proof=proof/1000000
proof = pd.to_datetime(proof, unit='s')
btc_options_call['expiration'] = proof
proof=btc_options_call['timestamp'].to_numpy(dtype="double")
proof=proof/1000000
proof = pd.to_datetime(proof, unit='s')
btc_options_call['timestamp'] = proof


#ds=btc_options_call
ds= pd.read_csv('btc_options_call_dateformat.csv')
print( ds.head() )
#ds.to_csv(r'C:\Users\gema_\OneDrive\Escritorio\Tesis\btc_options_call_dateformat.csv', index = False)
ds.drop('Unnamed: 0', axis=1, inplace=True)
ds.drop('exchange', axis=1, inplace=True)
ds.drop('symbol', axis=1, inplace=True)
ds.drop('local_timestamp', axis=1, inplace=True)
ds.drop('last_price', axis=1, inplace=True)
ds.drop('bid_price', axis=1, inplace=True)
ds.drop('bid_amount', axis=1, inplace=True)
ds.drop('ask_price', axis=1, inplace=True)
ds.drop('ask_amount', axis=1, inplace=True)
ds.drop('ask_iv', axis=1, inplace=True)
ds.drop('mark_price', axis=1, inplace=True)
ds.drop('underlying_index', axis=1, inplace=True)
ds.drop('delta', axis=1, inplace=True)
ds.drop('gamma', axis=1, inplace=True)
ds.drop('vega', axis=1, inplace=True)
ds.drop('theta', axis=1, inplace=True)
ds.drop('rho', axis=1, inplace=True)
ds.drop('mark_iv', axis=1, inplace=True)
ds.drop('bid_iv', axis=1, inplace=True)
ds.drop('open_interest',axis=1,inplace=True)

ds['timestamp']=ds['timestamp'].str[:10]
ds['expiration']=ds['expiration'].str[:10]
ds['type'] = 1
ds['timestamp'] = pd.to_datetime(ds['timestamp'])
ds['expiration'] = pd.to_datetime(ds['expiration'])
ds['time'] = ds['expiration'] - ds['timestamp']
ds.drop('timestamp', axis=1, inplace=True)
ds.drop('expiration', axis=1, inplace=True)
ds['time'] = ds['time'].dt.days.astype('int16')
ds.to_csv(r'C:\Users\gema_\OneDrive\Escritorio\Tesis\btc_options_call_5Columns.csv', index = False)


'''==================== Dataset completed========================='''

'''======
Black-Scholes Model
call = 1
put = 0
========'''

def BS_model (S, K, T, r, sigma, vanilla = 1):
    d1 = (np.log( S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    if vanilla == 1:
        price = (S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    if vanilla == 0:
        price = (K * np.exp(-r * T) * norm.cdf(-d2)-S * norm.cdf(-d1))
    return price

'''======
Getting the prices of BS model 
========'''

def get_bsm_price(data,sigma):
    #create the list
    a=[]    
    #Get tensor size
    sizeT = list(data.size())[0]  
    for i in range(sizeT):
        S = data[i][2].item() #
        K = data[i][1].item() #
        T = data[i][3].item()/365 #
        r = 0.0348 #https://www.buybitcoinworldwide.com/es/indice-de-volatilidad/
        call_price = BS_model(S,K,T,r,sigma)
        a.append(call_price)
    #Convert to array then tensor
    arrayOut = np.array(a)
    outputData = torch.from_numpy(arrayOut)
    return a
'''======
A sequential container. 
Modules will be added to it in the order they are passed in the constructor. 
Alternatively, an ordered dict of modules can also be passed in.
https://pytorch.org/docs/stable/nn.html

hid_1 = 20
hid_2 = 80 #Number of hidden layers
model = nn.Sequential(
    nn.Linear(1,30),
    nn.Linear(30,1)
)
========'''

'''===Black Scholes Model Execution'''

boolean_option = ds['time']!=0
ds=ds[boolean_option]

sigma_value = 0.035 
data_tensor = torch.tensor(ds.values)
start = time.time()
bsm_prices=get_bsm_price(data_tensor,sigma=sigma_value)
bsm_prices_df= pd.DataFrame(bsm_prices,columns=['BSM_PRICE'])
print(bsm_prices)
ds.to_csv(r'C:\Users\gema_\OneDrive\Escritorio\Tesis\data_tensor.csv', index = False)
bsm_prices_df.to_csv(r'C:\Users\gema_\OneDrive\Escritorio\Tesis\bsm_prices_fromdatatensor.csv', index = False)


























