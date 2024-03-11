import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os


import pandas as pd
import numpy as np
import typing
import matplotlib.pyplot as plt

import tensorflow as tf
import keras
from keras import layers
from keras import ops
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input, TimeDistributed, Activation, Dropout
from sklearn.model_selection import train_test_split

os.environ["KERAS_BACKEND"] = "tensorflow"

class MyLSTM:
    def __init__(self, variables, target, test_size = 0.5, val_size = 0.2):

        self.variables = variables
        self.target = target
        self.test_size = test_size
        self.valid_size = val_size
        self.model = self.get_model()

    def split_data(self, random_state=42):
        X_train_val, X_test, y_train_val, y_test = train_test_split(self.variables, self.target, test_size=self.test_size, random_state=random_state)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=self.valid_size, random_state=random_state)

        return X_train, y_train, X_val, y_val, X_test, y_test

    def get_model(self):
        

        model = Sequential()
    
        # hidden layer 1
        model.add(LSTM(100, 
                    input_shape=(50,1), 
                    return_sequences=True))
        model.add(Dropout(0.2))

        # hidden layer 2
        model.add(LSTM(100))
        model.add(Dropout(0.2))

        # output layer
        model.add(Dense(1))
        model.add(Activation("linear"))


        model.compile(optimizer='adam', loss='mean_squared_error')
        
        return model
    
    
    def train_model(self):
        x_train, y_train, x_val, y_val, x_test, y_test = self.split_data()

        callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=2,verbose=0)]
        
        self.model.fit(x_train, y_train, epochs=20, batch_size=16, verbose=1,validation_split=0.05, callbacks=callbacks)

        #history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), verbose=1)

        #loss = model.evaluate(X_test, y_test)

        #predictions = model.predict(X_test)




















        
        #scaler = MinMaxScaler()
        #self.data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns, index=data.index)
        #print(target.head())
        #print(self.data.head())
        #self.target = pd.DataFrame(scaler.fit_transform(target), columns=target.columns, index=target.index)
        #num_time_steps = self.data.shape[0]
        #num_train, num_val = (
        #    int(num_time_steps * train_size),
        #    int(num_time_steps * val_size),
        #)
        #train_array = self.data[:num_train]
        #mean, std = train_array.mean(axis=0), train_array.std(axis=0)
        #self.train_data = (train_array - mean) / std
        #self.valid_data = (self.data[num_train : (num_train + num_val)] - mean) / std
        #self.test_data = (self.data[(num_train + num_val) :] - mean) / std
        #self.train_target = (train_array - mean) / std
        #self.valid_target = (self.target[num_train : (num_train + num_val)] - mean) / std
        #self.test_target = (self.target[(num_train + num_val) :] - mean) / std
        #print(f"train set size: {self.train_data.shape}")
        #print(f"validation set size: {self.valid_data.shape}")
        #print(f"test set size: {self.test_data.shape}")


