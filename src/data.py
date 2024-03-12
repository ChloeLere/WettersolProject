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
            continue
        current = pd.read_csv(folder_path + filename).drop(columns=["Kvalitet"])
        current['Datum'] = pd.to_datetime(current['Datum'])
        current = current.groupby(current['Datum'].dt.date)['Solskenstid'].sum().reset_index()
        station_number = re.findall(r'\d+', filename)[0]
        merged_data = pd.merge(merged_data, current, on='Datum', how='outer', suffixes=('', f'_{station_number}'))
     
    merged_data['Datum'] = pd.to_datetime(merged_data['Datum'])
    columns_to_average = merged_data.select_dtypes(include=['int', 'float']).columns
    merged_data["AverageSolskenstid"] = merged_data[columns_to_average].mean(axis=1)
    data = merged_data[['Datum', "AverageSolskenstid"]]
    return(data)
    

def get_weather_table():
    data_lufttemperatur = weather_average_by_day("../data/wettersol/smhi_data_2022-today/parameter_2/", "AverageLufttemperatur")
    data_nederbordsmangd = weather_average_by_day("../data/wettersol/smhi_data_2022-today/parameter_5/", "AverageNederbördsmängd")
    data_snodjup = weather_average_by_day("../data/wettersol/smhi_data_2022-today/parameter_8/", "AverageSnödjup", ["Kvalitet", "Tid (UTC)"])
    data_solskenstid = weather_parameter_10("../data/wettersol/smhi_data_2022-today/parameter_10/")

    data = pd.merge(data_lufttemperatur, data_nederbordsmangd, on='Datum', how='outer')
    data = pd.merge(data, data_snodjup, on='Datum', how='outer')
    data = pd.merge(data, data_solskenstid, on='Datum', how='outer')

    return(data)

def get_capacity_from_zip(zip_code):
    capacities = get_data("../data/installed_capacity.csv")
    capacities = capacities.rename(columns={"56392,0": "Zip_Code", "15,21": "Capacity"})
    capacities["Capacity"] = capacities["Capacity"].str.replace(',', '.').astype(float)
    capacity = capacities[capacities["Zip_Code"] == int(zip_code)]["Capacity"].unique()
    return capacity[0]

def get_energy_produced_panel(zip_code):
    data_company = get_data("../data/" + zip_code + ".csv")
    data_company.rename(columns={'Date': 'Datum'}, inplace=True)
    data_company['Datum'] = pd.to_datetime(data_company['Datum'])
    data_company['EnergyProduced_Panel1'] = data_company['EnergyProduced_Panel1'].str.replace(',', '.').astype(float)
    if (len(data_company.columns) > 2):
        data_company['EnergyProduced_Panel2'] = data_company['EnergyProduced_Panel2'].str.replace(',', '.').astype(float)
        data_company['EnergyProduced'] = data_company['EnergyProduced_Panel1'] + data_company['EnergyProduced_Panel2']
        data_company = data_company.drop(['EnergyProduced_Panel1', 'EnergyProduced_Panel2'], axis=1)
    else:
        data_company.rename(columns={'EnergyProduced_Panel1': 'EnergyProduced'}, inplace=True)
    
    return data_company

def get_radiation():
    data_radiation = get_data("../data/solar_radiation.csv")
    data_radiation.replace(-999, float("NaN"), inplace=True)
    data_radiation['Datum'] = pd.to_datetime(data_radiation['Datum'])
    data_radiation['UV-irradiation'].interpolate(method='linear', inplace=True)
    return data_radiation

def get_table(zip_code):
    data_weather = get_weather_table()
    data_company = get_energy_produced_panel(zip_code)
    data_radiation = get_radiation()
    data = pd.merge(data_weather, data_radiation, on='Datum', how='inner')
    data = pd.merge(data, data_company, on='Datum', how='inner')
    data = data.dropna()
    data["Capacity"] = get_capacity_from_zip(zip_code)
    data = data.set_index('Datum')
    return data

# Split the dataframe between:
# X: Variables
# y: Target Prediction
def split_xy(dataframe, target_col_name):
    x = dataframe.drop(columns=target_col_name)
    y = dataframe[target_col_name]
    return x, y

def time_split(variables, target, offset=0, training_size=100, testing_size=50):
    if offset + training_size + testing_size > len(target):
        raise ValueError("offset + training_size + testing_size > len(self.target)")
    X_train = variables[offset:offset + training_size]
    y_train = target[offset:offset + training_size]
    X_test = variables[offset + training_size:offset + training_size + testing_size]
    y_test = target[offset + training_size:offset + training_size + testing_size]
    return X_train, y_train, X_test, y_test