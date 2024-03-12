import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
import os
from data import time_split, split_xy
from sklearn.model_selection import TimeSeriesSplit

class MyAutoregressive:
    def __init__(self, data, test_size = 0.5, val_size = 0.2, lags = 5):
        self.test_size = test_size
        self.valid_size = val_size
        self.lags = lags
        self.data = data

        variables, target = split_xy(data, "EnergyProduced")
        self.target = target
        self.variables = variables

    def train(self):

        x_train, y_train, x_test, y_test = time_split(self.variables, self.target, testing_size=7)
        
        model = AutoReg(y_train, lags=5, exog=x_train)
        model_fit = model.fit()
        print('Coefficients: %s' % model_fit.params)
        predictions = model_fit.predict(start=len(x_train), end=len(x_train) + len(x_test) - 1, exog_oos=x_test)

        for i in range(len(predictions)):
            print('predicted=%f, expected=%f' % (predictions[i], y_test[i]))

    def train_2(self):
        for i in range(1, self.lags+1):
            self.data[f'lag_{i}'] = self.data['EnergyProduced'].shift(i)
        self.data.dropna(inplace=True)
        variables, target = split_xy(self.data, "EnergyProduced")
        self.target = target
        self.variables = variables

        x_train, y_train, x_test, y_test = time_split(self.variables, self.target, testing_size=7)
        
        model = AutoReg(y_train, lags=self.lags)
        model_fit = model.fit()
        print('Coefficients: %s' % model_fit.params)

        y_pred = model_fit.predict(start=len(x_train), end=len(x_train) + len(x_test) - 1, dynamic=False)  # Make predictions
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5
        print(f'Root Mean Squared Error (RMSE): {rmse}')

        #predictions = model_fit.predict(start=len(x_train), end=len(x_train) + len(x_test) - 1, exog_oos=x_test)    
        for i in range(len(y_pred)):
            print('predicted=%f, expected=%f' % (y_pred[i], y_test[i]))