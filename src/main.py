#!/usr/bin/env python3

import sys
from data import get_table, split_xy
from visualization import Visualization
import pandas as pd

def main(argv):
    zip_code = 18645
    if len(argv) > 1:
        zip_code = argv[1]
    data: pd.DataFrame = get_table(str(zip_code))
    visualizations = Visualization(data)
    visualizations.visualization_energy()
    visualizations.visualization_every_column()
    visualizations.visualization_with_weather()
    visualizations.visualization_with_radiation()
    #variables, target = split_xy(data, "EnergyProduced")
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))