#!/usr/bin/env python3

import sys
from data import get_table, split_xy, plot_metrics, visualize_data
import pandas as pd
from lstm import MyLSTM
from autoregressive import MyAutoregressive
from randomForest import MyRandomforest
import warnings
warnings.filterwarnings("ignore")

def main(argv):
    zip_code = 55448
    if len(argv) > 1:
        zip_code = argv[1]
    data: pd.DataFrame = get_table(str(zip_code))

    # Visualization of the data (uncomment if needed) :
    #visualize_data(data)
    
    # Split the data :
    variables, target = split_xy(data, "EnergyProduced")

    # Long Short Term Memory :
    lstm = MyLSTM(variables, target, 5, 400)
    lstm.train()
    lstm.evaluate_loss()
    print("=============================Long short term memory=============================")
    lstm.predict()
    lstm_metrics = lstm.metrics()

    # Autegressive :
    print("=================================AutoRegressive=================================")
    ar = MyAutoregressive(data, lags=5)
    ar.train(0, 400, 5)
    ar.predict(display=True)
    ar_metrics = ar.metrics()

    #Random forest :
    print("=================================Random  Forest=================================")
    rf = MyRandomforest(variables, target)
    rf.train()
    rf.predict(display=True)
    rf_metrics = rf.metrics()

    #Plot the result :
    plot_metrics([ar_metrics, lstm_metrics, rf_metrics], ["AutoRegression", "LSTM", "RandomForest"])

    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))