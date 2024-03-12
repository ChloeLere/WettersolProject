#!/usr/bin/env python3

import sys
from data import get_table, split_xy
from visualization import Visualization
import pandas as pd
from lstm import MyLSTM
from autoregressive import MyAutoregressive

def main(argv):
    zip_code = 18645
    if len(argv) > 1:
        zip_code = argv[1]
    data: pd.DataFrame = get_table(str(zip_code))

    # Visualization of the data :
    #visualizations = Visualization(data)
    #visualizations.visualization_energy()
    #visualizations.visualization_every_column()
    #visualizations.visualization_with_weather()
    #visualizations.visualization_radiation()
    
    # Split the data :
    variables, target = split_xy(data, "EnergyProduced")

    # Long Short Term Memory :
    #lstm = MyLSTM(variables, target)
    #lstm.train_model()

    # Autegressive

    ar = MyAutoregressive(data, lags=5)
    ar.train()

    #Random forest :

    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))