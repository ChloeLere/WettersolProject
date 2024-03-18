import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from data import time_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

class MyRandomforest:
    def __init__(self, variables, target, test_size = 5, training_size = 400):
        self.mse = 0.0
        self.rmse = 0.0
        self.mae = 0.0
        self.r_squared = 0.0

        self.x_train, self.y_train, self.x_test, self.y_test = time_split(variables, target, training_size=training_size, testing_size=test_size)
        self.model = self.create_model()

    def param_tuning(self):
        param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
        }
        
        grid_search = GridSearchCV(self.model, param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(self.x_train, self.y_train)
        best_params = grid_search.best_params_
        print("Best hyperparameters:", best_params)

    def create_model(self, max_depth = 10, min_samples_split = 2, n_estimators = 200):
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
        return model

    def train(self):
        self.model.fit(self.x_train, self.y_train)
    
    def predict(self, display=False):
        predictions = self.model.predict(self.x_test)
        date_test = self.x_test.index
        if display:
            print("Predictions :")
            for i in range(len(predictions)):
                print('predicted=%f, expected=%f, for the date=%s' % (predictions[i], self.y_test[i], date_test[i].strftime("%Y-%m-%d")))

        self.mse = mean_squared_error(self.y_test, predictions)
        self.mae = mean_absolute_error(self.y_test, predictions)
        self.rmse = np.sqrt(self.mse)
        self.r_squared = r2_score(self.y_test, predictions)
    
    def cross_validation(self):
        cv_scores = cross_val_score(self.model, self.x_train, self.y_train, cv=5, scoring='neg_mean_squared_error')
        mean_cv_mse = -np.mean(cv_scores)
        print(f"Mean CV MSE: {mean_cv_mse:.4f}")
        
        n_splits = 5
        tscv = TimeSeriesSplit(n_splits=n_splits)

    def metrics(self):
        print("\nMetrics Random Forest")
        print("Mean Squared Error:", self.mse)
        print("Mean Absolute Error:", self.mae)
        print("Root Mean Squared Error:", self.rmse)
        print("R Squared score:", self.r_squared)
        return self.mse, self.mae, self.rmse, self.r_squared
