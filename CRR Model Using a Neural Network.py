# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 19:23:28 2021

@author: gema_
"""

import torch
import torch.nn as nn
import numpy as numpy
import torch.nn.functional as F
import matplotlib.pyplot as plt

#traget function
a = 1.
b = 2.
f = lambda x: a*x+b

#test 
(2.)

#model nn.Linear
in_dim = 1
out_dim = 1

model = nn.Sequential(nn.Linear(in_dim,50), 
                      nn.ReLU(),
                      nn.Linear(50,out_dim) 
                      )

#Loss function
criterion = nn.MSELoss()
#Optimizer
learning_rate = 0.001
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)
#training data
batch_size = 1000
x_train=torch.randn(batch_size,1)
y_train=f(x_train)


    #type call/put 
    #n = steps/height of bonimial tree
    #T = time until maturity
    #S = base price
    #K=strike price
    #r=interest
    #q=dividend
    #sigma = volatility
    #tree = show option free if True doesn´t show if False
def CRRAmericanOption(type,n,T,S,K,r,sigma,q,tree):
    dt = T/n #delta t for each step
    u = numpy.exp(sigma*numpy.sqrt(dt)) # price multiplier if it goes up
    d = 1./u # price multiplier if it does down
    p = (numpy.exp((r-q)*dt)-d) / (u-d) #formula for calculating probability for each price
    
    #Binomial tree
    #constructing the tree
    binomial_tree = numpy.zeros([n+1,n+1])
    
    #initializing the tree
    for i in range (n+1):
        for j in range (i+1):
            binomial_tree[j,i] = S*(d**j)*u**(i-j)
#        print("bonimial tree {}".format(i))
#        print(binomial_tree)
    #Exercise tree
    #constructing the tree
    exercise_tree = numpy.zeros([n+1,n+1])
    #print(exercise_tree)
    #initializing the tree
    for i in range (n+1):
        for j in range (i+1):
            exercise_tree[j,i] = K
    #print("Exercise tree:")
    #print(exercise_tree)      
    #option value
    option = numpy.zeros([n+1,n+1])
    #call option value is Max[(Sn+K),0]
    if type == 'call':
        option[:n]= numpy.maximum(numpy.zeros(n+1),binomial_tree[:,n]-exercise_tree[:,n])
        #print(option[:,n])
    if type == 'put':
        option[:n]= numpy.maximum(numpy.zeros(n+1),exercise_tree[:,n]-binomial_tree[:,n])
        print(option[:,n])
    #calculating the price at t=0
    for i in numpy.arange(n-1,-1,-1):
        for j in numpy.arange(0,i+1):
            option[j,i]=numpy.exp(-r*dt)-(p*option[j,i+1]+(1-p)+option[j+1,i+1])
    if type == 'call':
        option[:,n]= numpy.maximum(option[:,n],binomial_tree[:,n]-exercise_tree[:,n])
        #print(option[:,n])
    if type == 'put':
        option[:,n] = numpy.maximum(option[:,n], exercise_tree[:,n]-binomial_tree[:,n])
        print(option[:,n])
    #return value
    if tree:
        print(numpy.matriz(option))
        #print("Option Value: ")
        #print(option)
        return option[0,0]
    else:
        #print("Option Value: ")
        #print(option)
        return option[0,0]

#train model
num_epochs = 1000
for epoch in range(num_epochs):
    #forward pass
    outputs = model(x_train)
    loss = criterion (outputs, y_train.float())
    
    #backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    #if epoch == 0 or (epoch+1)%50 == 0:
     #   print('Epoch [{}/{}], Loss: {: .4f}'.format(epoch+1,num_epochs,loss.item()))
#training data
#training size per epoch
#print(len(outputs))
batch_size = 100
#creating a list
a = []

#generate randomize data
x_train = torch.randn(batch_size,1)

#run loop on trining data to creaat output array
for i in range(batch_size):
    y = CRRAmericanOption('put', 10 , 100, 200, x_train[i], 0.05, 0.11, 0.1, False)
    a.append(y)

#transform list to array
b = numpy.array(a)

#transform array into tensor set
y_train = torch.from_numpy(b)

#plt.scatter(x_train.detach().numpy(),outputs.detach().numpy(),label='true')
#y_train.detach().numpy()
plt.scatter(x_train.detach().numpy(),b,label='pred')
