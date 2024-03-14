#!/usr/bin/env python3

import sys
from data import get_table, split_xy, plot_metrics, visualize_data
import pandas as pd
from lstm import MyLSTM
from autoregressive import MyAutoregressive
from randomForest import MyRandomforest

def main(argv):
    zip_code = 18645
    if len(argv) > 1:
        zip_code = argv[1]
    data: pd.DataFrame = get_table(str(zip_code))

    # Visualization of the data :
    #visualize_data(data)
    
    # Split the data :
    variables, target = split_xy(data, "EnergyProduced")

    # Long Short Term Memory (need to call everything in this order):
    lstm = MyLSTM(variables, target, 5, 400)
    lstm.train()
    lstm.evaluate_loss()
    lstm_metrics = lstm.metrics()

    # Autegressive
    ar = MyAutoregressive(data, lags=5)
    ar.train(0, 400, 5)
    ar_metrics = ar.metrics()

    #Random forest :
    rf = MyRandomforest(variables, target, lags=5)
    rf.train_model()

    plot_metrics([ar_metrics, lstm_metrics], ["AutoRegression", "LSTM"])

    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))