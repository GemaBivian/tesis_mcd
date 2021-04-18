# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 21:58:47 2021

@author: gema_
"""

import torch 
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
import time

'''======
option class init
======'''
class VanillaOption:
    def __init__(
            self,
            otype =1, # 1:'call'/ -1:'put'
            strike = 110.,
            maturity = 1.,
            market_price = 10.):
        self.otype = otype #put or call
        self.strike = strike #strike k
        self.maturity = maturity #maturity T
        self.market_price = market_price # this will be used for calibration
        
        def payoff(self , s): #s:exercise price
            otype = self.otype
            k = self.strike
            maturity = self.maturity
            return np.max([0,(s-k)*otype])
'''======
Gbm class
======'''
class Gbm:
    def __init__(self,
                 init_state = 100.,
                 drift_ratio = .0475,
                 vol_ratio = .2):
        self.init_state = init_state
        self.drift_ratio = drift_ratio
        self.vol_ratio = vol_ratio

'''======
Black-Scholes Merton formula
======'''
def bsm_price(self, vanilla_option):
    s0 = self.init_state
    sigma = self.vol_ratio
    r = self.drift_ratio
    
    otype = vanilla_option.otype
    k = vanilla_option.strike
    maturity = vanilla_option.maturity
    d1= 1/(sigma*np.sqrt(maturity))*(np.log(s0/k)+(r+np.power(sigma,2)/2)*(maturity))
    d2= 1/(sigma*np.sqrt(maturity))*(np.log(s0/k)+(r-np.power(sigma,2)/2)*(maturity))
    
    return (otype * s0 * ss.norm.cdf(otype*d1) #line break need parenthesis
            - otype * np.exp(-r*maturity)*k*ss.norm.cdf(otype*d2))
Gbm.bsm_price = bsm_price

'''======
Get BSM prices given an option and Tensor
======'''
def prices_bsm(self,vanilla_option,data):
    #create the list
    a=[]
    
    #Get tensor size
    sizeT = list(data.size())[0]
    
    for i in range(sizeT):
        self.init_state = data[i].item()
        callPrice = Gbm.bsm_price(vanilla_option)
        a.append(callPrice)
    #Convert to array then tensor
    arrayOut = np.array(a)
    outputData = torch.from_numpy(arrayOut)
    return outputData

Gbm.prices_bsm = prices_bsm


'''Definition of a Function to Calculate a Group of BSM Prices Given a Tensor'''
def f(s, k=90):
    gbm = Gbm(init_state=s)
    option = VanillaOption(strike = k)
    return gbm.bsm_price(option)

batch_size = 63
x_list = np.linspace(0,200,batch_size)
y_list = np.array([f(x) for x in x_list])

'''Definition of a function to Calculate BSM Price given a Single Underlying and
Strike Price And Creation of a Random List of Strike Prices For Model Training'''
#model
#nn.Linear
H1 = 80; H2=20 #number of hidden layer
model = nn.Sequential(
    nn.Linear(1,H1),
    nn.Sigmoid(),
    nn.Linear(H1,H2),
    nn.Sigmoid(),
    nn.Linear(H2,2),
    nn.Sigmoid(),
    nn.Linear(2,1)
    )

'''Definition of a Neural Network Model'''
#loss function
criterion = nn.MSELoss()

'''Definition of the Loss Function as Mean Squared Error'''
#Optimizer 
learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate,momentum=0.7)

''''Definition of the Method of Optimization for the Network'''
batch_size = np.size(x_list)
x_train0 = torch.from_numpy(x_list).reshape(batch_size,1).float()
y_train0 = torch.from_numpy(y_list).reshape(batch_size,1).float()


'''Creation of Tensors of Training Data'''
#normalization

def linear_transform(xx,l = 0, u = 1 ):
    M = torch.max(xx)
    m = torch.min(xx)
    return (u-l)/(M-m)*(xx-m)+l,m,M,l,u
x_train,x_m,x_M,x_l,x_u = linear_transform(x_train0,-1,1)
y_train,y_m,y_M,y_l,y_u = linear_transform(y_train0,0,1)
'''Definition of the Linear Transform Function'''
# train the model
num_epochs = 5000

for epoch in range(num_epochs):
    
    #forward pass
    outputs = model(x_train.float())
    loss = criterion(outputs, y_train.float())
    #backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch == 0 or (epoch+1)%50 == 0:
        print('Epoch [{}/{}], Loss: {: .4f}'.format(epoch+1,num_epochs,loss.item()))
        print(outputs[1:10])


'''Loss Per Epoch while Training the Model'''
def learnedfun(x):
    out = (1-(-1))/(x_M-x_m)*(x-x_m)+(-1.)
    out = model(out)
    out = (y_M - y_m)*out+y_M
    return out

y_pred = learnedfun(x_train0)


'''Function for Linearly Transforming Model Output'''
#Test with training data
plt.scatter(x_train0.detach().numpy(), y_train0.detach().numpy(), label = 'true')
plt.scatter(x_train0.detach().numpy(),y_pred.detach().numpy(),label='pred')

plt.legend()
plt.show()


'''Testing of the Model Versus the Training Data. See Figure 12b for an Example
of Trained Data'''
#Test the model with random data
#Generate random data

x_0 = np.random.randint(0,200,batch_size)
#get bsm prices as determined by the formula

y_0 = np.array([f(x) for x in x_0])
#Transform to tensor
x_0 = torch.from_numpy(x_0).reshape(batch_size,1).float()
y_0 = torch.from_numpy(y_0).reshape(batch_size,1).float()

#normalize
x_, x_m, x_M, x_l, x_u = linear_transform(x_0)
y_, y_m, y_M, y_l, y_u = linear_transform(y_0)



#plot x_ versus formula prices
plt.scatter(x_0.detach().numpy(), y_0.detach().numpy(),label = "true")

#get BSM prices as determined by the model
y_pred = learnedfun(x_0)

# plot x_ versus the model prices
plt.scatter(x_0.detach().numpy(), y_pred.detach().numpy(), label="pred")

plt.legend()


'''Setup-Optimizer Code'''
#Best set up for least loss
bestLoss = 100
bestNeurons = 0
BestLayers = 0
bestTimeForLoss = 1000000 # i dont know what max_int equivalent is 

#Best set up fopr least time within < 0.001
neuronsBestTime= -1
layersBestTime = -1 
lossBestTime=1000000 #same as the above

#temporary variables
tempTime = 0
tempLoss =0

maxNeuron= 20
maxLayer = 8
#main code/model
for i in range(2,maxNeuron+1): #loop through each # of neuron 
	for j in range(2,maxLayer+1): #loop through each $ of layersBestTime
		model = nn.Sequential(
			nn.Linear(1,i),
			nn.Sigmoid(),
			nn.Linear(i,i),
			#loop for the layers inside
			#need help fixing this
			#for k in range(2,j)
			#nn.linear(i,i)
			nn.Linear(i,1)
		)
		
		#run the rest of the code with the current neurons and layers set up
		#start the timer
		start = time.time()
		#train the model
		for epoch in range (10000):
			#fordward pass
			outputs = model (x_train)
			loss =  criterion(outputs, y_train)
			#backward and optimize
			optimizer.zero_grad()            
			loss.backward()
			optimizer.step()


print(outputs)

#plot x_ versus formula prices

plt.scatter(x_train.numpy(), y_train.detach().numpy(), label = "true")

#get BSM prices as determined by the model
#y_pred = learnedfun(x_0)

# plot x_ versus the model prices
plt.scatter(x_train.numpy(), outputs.detach().numpy(), label="pred")

plt.legend()









