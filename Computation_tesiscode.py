# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 23:24:38 2021

@author: gema_
"""
#Approximating a Linear AND Non-Linear Function Using a Neural Network
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import math
#traget function
a = 1.
b = 2.
#f = lambda x: a*x+b
f_exp = lambda x: math.exp(x)
#test 
print(f_exp(2.))

#model nn.Linear
in_dim = 1
out_dim = 1
model = nn.Sequential(nn.Linear(in_dim,50), 
                      #nn.ReLU(),
                      nn.Linear(50,out_dim) 
                      )
#Loss function
loss_criteria = nn.MSELoss()
#Optimizer
learning_rate = 0.001
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)
#training data
x_train=torch.randn(1000,1)
y_train=f_exp(x_train)


#train the model
num_epochs = 500 #épocas el algoritmo trabaja con el dataset completo

for epoch in range(num_epochs):
    #fordward pass
    outputs = model(x_train)
    loss = loss_criteria(outputs, y_train)
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
# -----------
    



