# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 21:50:12 2021

@author: gema_
"""

#Importing Libraries and csv
import pandas as pd
import time 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn import preprocessing
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
ds.drop('AKA', axis=1, inplace=True)
ds.drop('Flags', axis=1, inplace=True)
ds.drop('T1OpenInterest', axis=1, inplace=True)
ds.drop('Type', axis=1, inplace=True)
#train test split
train_ds = ds.sample(frac=0.8,random_state=0)
test_ds = ds.drop(train_ds.index)

normed_train_data = preprocessing.normalize(train_ds)
normed_test_data = preprocessing.normalize(test_ds)
normed_train_data = pd.DataFrame(normed_train_data)
normed_test_data = pd.DataFrame(normed_test_data)

normed_train_data = normed_train_data.rename(columns={0:'UnderlyingPrice',1:'Strike',2:'IVMean',3:'AveragePrice',4:'TimeToMaturity'})
normed_test_data = normed_test_data.rename(columns={0:'UnderlyingPrice',1:'Strike',2:'IVMean',3:'AveragePrice',4:'TimeToMaturity'})
#Defining neural network architecture
def NN_model():
	model = keras.Sequential([layers.Dense(64, activation='linear', input_shape=[len(train_ds.keys())]),layers.Dense(64, activation='relu'),layers.Dense(64, activation='relu'),layers.Dense(64, activation='relu'),layers.Dense(1, activation='relu')])
	optimizer = tf.keras.optimizers.Adam(learning_rate=0.004)
	model.compile(loss='mae', optimizer=optimizer, metrics=['mae'])
	return model
model=NN_model()

#training the model using training set
tf.debugging.set_log_device_placement(True)
with tf.device('/device:GPU:0'):
	history = model.fit(normed_train_data, normed_train_data["Strike"], batch_size=128, epochs= 55 , validation_split = 0.2, verbose=0)
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
#evaluating the model using t est set and counting time to compute
start = time.time()
loss, mae = model.evaluate(normed_test_data, normed_test_data["Strike"], verbose=0)
end = time.time()

print(history.history)
