from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from data import time_split, split_xy

class MyAutoregressive:
    def __init__(self, data, test_size = 0.5, val_size = 0.2, lags = 5):
        self.test_size = test_size
        self.valid_size = val_size
        self.lags = lags
        self.data = data

        variables, target = split_xy(data, "EnergyProduced")
        self.target = target
        self.variables = variables

        self.mse = 0.0
        self.rmse = 0.0
        self.mae = 0.0
        self.r_squared = 0.0

    def clamp_negative(self, predictions):
        for i in range(len(predictions)):
            if predictions[i] < 0:
                predictions[i] = 0
        return predictions

    def train(self, offset=0, training_size=320, testing_size=5):
        self.XYs = time_split(self.variables, self.target, offset, training_size, testing_size)
        x_train, y_train, _, _ = self.XYs
        
        self.model = AutoReg(y_train, lags=self.lags, exog=x_train)
        self.model_fit = self.model.fit()

    def predict(self, display=False):
        x_train, _, x_test, y_test = self.XYs
        predictions = self.model_fit.predict(start=len(x_train), end=len(x_train) + len(x_test) - 1, exog_oos=x_test)
        predictions = self.clamp_negative(predictions)

        if display:
            for i in range(len(predictions)):
                print('predicted=%f, expected=%f' % (predictions[i], y_test[i]))
        
        self.mse = mean_squared_error(y_test, predictions)
        self.rmse = self.mse ** 0.5
        self.r_squared = r2_score(y_test, predictions)
        self.mae = mean_absolute_error(y_test, predictions)

    # get the metrics of the last train run
    def metrics(self):
        print("=================================AutoRegressive=================================")
        print("Mean Squared Error:", self.mse)
        print("Mean Absolute Error:", self.mae)
        print("Root Mean Squared Error:", self.rmse)
        print("R Squared score:", self.r_squared)
        return self.mse, self.mae, self.rmse, self.r_squared

    # don't use
    def train_2_unused(self):
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