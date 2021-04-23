# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 17:30:15 2021

@author: gema_
"""
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import math

ds=pd.read_csv(r'BTC_OPTIONS_VERSION2.csv')
ds.dtypes

ds['time_create'] = pd.to_datetime(ds['time_create'])
ds['time_expire'] = pd.to_datetime(ds['time_expire'])
ds['time'] = ds['time_expire'] - ds['time_create']
ds['time'] = ds['time'].dt.days.astype('int16')
boolean_option = ds['call']==1
ds = ds[boolean_option]
boolean_option = ds['time']>0
ds = ds[boolean_option]

a = 1.
b = 2.
#f = lambda x: a*x+b
f_exp = lambda x: math.exp(x)

#model nn.Linear
in_dim = 9
out_dim = 9
model = nn.Sequential(nn.Linear(in_dim,50), 
                      nn.ReLU(),
                      nn.Linear(50,out_dim) 
                      )
#Loss function
loss_criteria = nn.MSELoss()
#Optimizer
learning_rate = 0.001
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)
#1:price,3:index_price,10:time_expire, 11:date, 7:strike, 12:irate, 13:price_usd, 18:call,20:time
data_options = ds.iloc[:,[1,3,7,10,11,12,13,18,20]]
#training data
train_ds = data_options.sample(frac=0.8,random_state=0)
train_options = train_ds
test_ds = data_options.drop(train_ds.index)

#train the model
num_epochs = 500 #épocas el algoritmo trabaja con el dataset completo

for epoch in range(num_epochs):
    #fordward pass
    outputs = model(train_ds)
    loss = loss_criteria(outputs, train_ds)
    #backward and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()    
    if (epoch+1) % 50 == 0:
        print('Epoch [{}/{}], Loss {:.4f}'.format(epoch+1,num_epochs,loss.item()))    

#test
x_ = torch.randn(50,1)
y_ = f_exp(x_)
y_pred = model(x_)

plt.scatter(x_.detach().numpy(),y_.detach().numpy(),label='true')
plt.scatter(x_.detach().numpy(),y_pred.detach().numpy(),label='pred')
plt.legend()

