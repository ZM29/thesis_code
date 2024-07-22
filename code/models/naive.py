"""
This script returns the naive forecaster's predictions
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def load_data(file_path):
    """
    Function to change load data and change format.
    """
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.set_index(['Date'])
    closing_data = data.iloc[:,:2]
    
    return closing_data

def stationary(data):
    """
    Function to make the data stationary by using the ADF test. If a feature (series) is non-stationary, 
    the function is repeated until the series is stationary.
    """
    stationary_data = pd.DataFrame()
    
    for column in data.columns:
        series = data[column]

        #NAs need to be dropped to ensure results 
        result = adfuller(series.dropna())
        
        #results[1] is the P-value of the test
        if result[1] > 0.05:
            diff_series = stationary(pd.DataFrame(series.diff().dropna(), columns = [column]))
            stationary_data = pd.concat([stationary_data, diff_series], axis = 1)
        else:
            stationary_data = pd.concat([stationary_data, series], axis = 1)

    return stationary_data

def naive_forecaster(data):
    """
    Naive forecaster which returns the last known value
    """
    return data[-1:]


file_path = r'datasets\numeric_data.csv'
data = load_data(file_path)
index_names = data['Index'].unique()

#looping over all the indices
for index_name in index_names:
    print(f"Evaluating index: {index_name}")

    #filtering the data
    index_data = data[data['Index'] == index_name]
    index_data = index_data.drop("Index", axis = 1)

    #making the data stationary
    stationary_data = stationary(index_data)

    _, test_data = train_test_split(stationary_data, test_size = 0.2, shuffle = False)

    #making the forecasts and computing MSE
    forecasts = [naive_forecaster(test_data[ : i + 1]) for i in range(len(test_data) - 1)]
    forecasts = np.array(forecasts).reshape(-1, 1)
    actual_values = test_data[1:].values.reshape(-1, 1)
    mse = mean_squared_error(actual_values, forecasts)


