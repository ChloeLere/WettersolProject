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
        self.x_train = self.scaler.transform(self.x_train)
        self.x_test = self.scaler.transform(self.x_test)
        self.model, self.generator = self.create_model()
        
    
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
        prediction = self.model.predict(to_test)
        print(prediction)
        print(self.y_test[0])


    def metrics(self):
        to_test = self.x_test.reshape((1, self.test_size, 6))
        predictions = np.array(self.model.predict(to_test)).flatten()
        true_y = np.array(self.y_test)
        mse = mean_squared_error(true_y, predictions)
        mae = mean_absolute_error(true_y, predictions)
        rmse = np.sqrt(mse)
        r_squared = r2_score(true_y, predictions)

        print("=============================Long short term memory=============================")
        print("Mean Squared Error:", mse)
        print("Mean Absolute Error:", mae)
        print("Root Mean Squared Error:", rmse)
        print("R Squared score:", r_squared)

        return mse, mae, rmse, r_squared
