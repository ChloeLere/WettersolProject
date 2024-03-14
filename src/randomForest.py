import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from data import time_split, split_xy
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

class MyRandomforest:
    def __init__(self, variables, target, test_size = 0.5, val_size = 0.2, lags = 5):
        self.test_size = test_size
        self.valid_size = val_size
        self.lags = lags
        self.target = target
        self.variables = variables

        self.mse = 0.0
        self.rmse = 0.0
        self.mae = 0.0
        self.r_squared = 0.0
    
    def param_tuning(self):
        pass

    def train_model(self):

        x_train, y_train, x_test, y_test = time_split(self.variables, self.target, testing_size=7)

        model = RandomForestRegressor(n_estimators=200, max_depth=20, min_samples_split=2, random_state=42)
        model.fit(x_train, y_train)
         
        predictions = model.predict(x_test)
        for i in range(len(predictions)):
            print('predicted=%f, expected=%f' % (predictions[i], y_test[i]))
            
        param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
        }
        
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(x_train, y_train)
        best_params = grid_search.best_params_
        print("Best hyperparameters:", best_params)
        
        # Evaluate model using cross-validation
        cv_scores = cross_val_score(model, x_train, y_train, cv=5, scoring='neg_mean_squared_error')
        mean_cv_mse = -np.mean(cv_scores)
        print(f"Mean CV MSE: {mean_cv_mse:.4f}")
        
        n_splits = 5
        tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # get the metrics of the last train run
    def metrics(self):
        print("=================================Random  Forest=================================")
        print("Mean Squared Error:", self.mse)
        print("Mean Absolute Error:", self.mae)
        print("Root Mean Squared Error:", self.rmse)
        print("R Squared score:", self.r_squared)
        return self.mse, self.mae, self.rmse, self.r_squared

# tuning ailleur
# tuning into new train
# good split
# remove useless params
# get metrics