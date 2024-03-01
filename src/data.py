import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import re
pd.set_option('display.max_rows', None) 
pd.set_option('display.max_columns', None) 

def get_data(filename):
    file_result = pd.read_csv(filename)
    return file_result

def plot_data(data, xlabel, ylabel, graph_name):
    plt.plot(data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(graph_name)
    plt.show()


def weather_average_by_day(folder_path, column_name_average, column_to_rm = ["Kvalitet"]):
    files = os.listdir(folder_path)
    first_round = True
    start_date = pd.Timestamp("2022-01-01")
    end_date = pd.Timestamp('2024-01-01')

    for filename in files:
        if first_round:
            merged_data = pd.read_csv(folder_path + filename).drop(columns=column_to_rm)
            first_round = False
            continue
        current = pd.read_csv(folder_path + filename).drop(columns=column_to_rm)
        station_number = re.findall(r'\d+', filename)[0]
        merged_data = pd.merge(merged_data, current, on='Datum', how='outer', suffixes=('', f'_{station_number}'))

    merged_data['Datum'] = pd.to_datetime(merged_data['Datum'])
    merged_data = merged_data[(merged_data['Datum'] >= start_date) & (merged_data['Datum'] <= end_date)]

    columns_to_average = merged_data.select_dtypes(include=['int', 'float']).columns
    merged_data[column_name_average] = merged_data[columns_to_average].mean(axis=1)
    data = merged_data[['Datum', column_name_average]]
    return (data)

def weather_parameter_10(folder_path):
    files = os.listdir(folder_path)
    first_round = True

    for filename in files:
        if first_round:
            merged_data = pd.read_csv(folder_path + filename).drop(columns=["Kvalitet"])
            first_round = False
            merged_data['Datum'] = pd.to_datetime(merged_data['Datum'])
            merged_data = merged_data.groupby(merged_data['Datum'].dt.date)['Solskenstid'].sum().reset_index()
            print(merged_data.head())
            continue
        current = pd.read_csv(folder_path + filename).drop(columns=["Kvalitet"])
        current['Datum'] = pd.to_datetime(current['Datum'])
        current = current.groupby(current['Datum'].dt.date)['Solskenstid'].sum().reset_index()
        station_number = re.findall(r'\d+', filename)[0]
        merged_data = pd.merge(merged_data, current, on='Datum', how='outer', suffixes=('', f'_{station_number}'))
     
    columns_to_average = merged_data.select_dtypes(include=['int', 'float']).columns
    merged_data["AverageSolskenstid"] = merged_data[columns_to_average].mean(axis=1)
    data = merged_data[['Datum', "AverageSolskenstid"]]
    return(data)
    

def get_table():
    weather_average_by_day("../data/wettersol/wettersol/smhi_data_2022-today/parameter_2/", "AverageLufttemperatur")
    weather_average_by_day("../data/wettersol/wettersol/smhi_data_2022-today/parameter_5/", "AverageNederbördsmängd")
    weather_average_by_day("../data/wettersol/wettersol/smhi_data_2022-today/parameter_8/", "AverageSnödjup", ["Kvalitet", "Tid (UTC)"])
    weather_parameter_10("../data/wettersol/wettersol/smhi_data_2022-today/parameter_10/")