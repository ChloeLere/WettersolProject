#!/usr/bin/env python3

import sys
from data import get_table 
from time_series_forcating import TimeSeriesForcasting

def main(argv):
    zip_code = 18645
    if len(argv) > 1:
        zip_code = argv[1]
    data = get_table(str(zip_code))
    tsf = TimeSeriesForcasting(data)
    tsf.visualization_energy()
    tsf.visualization_every_column()
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))