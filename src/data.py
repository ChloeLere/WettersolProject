import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def get_data(filename):
    file_result = pd.read_csv(filename)
    return file_result

def plot_data(data, xlabel, ylabel, graph_name):
    plt.plot(data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(graph_name)
    plt.show()
