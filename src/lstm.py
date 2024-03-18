from sklearn.preprocessing import MinMaxScaler
from data import time_split
from keras.preprocessing.sequence import TimeseriesGenerator

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

class MyLSTM:
    def __init__(self, variables, target, test_size = 5, training_size = 400):
        self.variables = variables
        self.target = target
        self.test_size = test_size
        self.x_train, self.y_train, self.x_test, self.y_test = time_split(variables, target, training_size=training_size, testing_size=test_size)
        self.scaler = MinMaxScaler()
        self.scaler.fit(self.x_train)
        self.date_test = self.x_test.index
        self.x_train = self.scaler.transform(self.x_train)
        self.x_test = self.scaler.transform(self.x_test)
        self.model, self.generator = self.create_model()
        self.mse = 0.0
        self.rmse = 0.0
        self.mae = 0.0
        self.r_squared = 0.0
    
    def create_model(self):
        n_features = 6
        n_input = self.test_size
        generator = TimeseriesGenerator(self.x_train, self.y_train, length=n_input, batch_size=1)
        model = Sequential()
        model.add(LSTM(100, activation='relu', input_shape=(n_input, n_features)))
        model.add(Dense(n_input))

        return model, generator
    
    def train(self, summarize=False, epochs=100, batch_size=32):
        self.model.compile(optimizer='adam', loss='mse')
        if summarize:
            self.model.summary()
        self.model.fit(self.generator, epochs=epochs, batch_size=batch_size, verbose=1)

        
    def evaluate_loss(self, display=False):
        loss_per_epoch = self.model.history.history['loss']
        if display:
            plt.plot(range(len(loss_per_epoch)),loss_per_epoch)
            plt.show()
        return loss_per_epoch
    
    def predict(self):
        to_test = self.x_test.reshape((1, self.test_size, 6))
        predictions = self.model.predict(to_test)
        print("Predictions :")
        for i in range(len(self.y_test)):
            print('predicted=%f, expected=%f, for the date=%s' % (predictions[0][i], self.y_test[i], self.date_test[i].strftime("%Y-%m-%d")))
        predictions = np.array(predictions).flatten()
        true_y = np.array(self.y_test)
        self.mse = mean_squared_error(true_y, predictions)
        self.mae = mean_absolute_error(true_y, predictions)
        self.rmse = np.sqrt(self.mse)
        self.r_squared = r2_score(true_y, predictions)

    def metrics(self):
        print("\nMetrics Long Short Term Memory :")
        print("Mean Squared Error:", self.mse)
        print("Mean Absolute Error:", self.mae)
        print("Root Mean Squared Error:", self.rmse)
        print("R Squared score:", self.r_squared)

        return self.mse, self.mae, self.rmse, self.r_squared
