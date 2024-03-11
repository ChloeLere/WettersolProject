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

        #callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=2,verbose=0)]
        #self.model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=1,validation_split=0.05, callbacks=callbacks)
        
        self.model.fit(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_val, y_val), verbose=1)

        loss = self.model.evaluate(x_test, y_test)

        predictions = self.model.predict(x_test)
 
