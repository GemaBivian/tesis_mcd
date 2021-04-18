# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 18:26:30 2021

@author: gema_
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
#traget function
a = 2.
b = 3.
#linear function
f_linear = lambda x: a*x+b
f_quad = lambda x: a*x*x+b
f_exp = lambda x: torch.exp(x)

functions = [f_linear,f_quad,f_exp]

#model 
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

for i in range(len(functions)):
    y_train=functions[i](x_train)
    plt.scatter(x_train.detach().numpy(),y_train.detach().numpy(),label='function {}'.format(i))
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
            print('Epoch [{}/{}] Loss {:.6f}'.format(epoch+1,num_epochs,loss.item())) 
    x_ = torch.randn(50,1)
    y_ = functions[i](x_)
    y_pred = model(x_)

    plt.scatter(x_.detach().numpy(),y_.detach().numpy(),label='true {}'.format(i))
    plt.scatter(x_.detach().numpy(),y_pred.detach().numpy(),label='pred {}'.format(i))
    plt.legend()
    plt.show() 
