"""
This script plots the index prices from 2010 onward.
"""

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt


def plot_index(data):
    index_rename = {'^GSPC': 'S&P 500', '^RUT': 'Russell 2000', '^IXIC': 'NASDAQ Composite', '^FTSE': 'FTSE 100', '^DJI': 'Dow Jones'}
    data['Index'] = data['Index'].replace(index_rename)

    #splitting the data
    ixic_dji_data = data[data['Index'].isin(['NASDAQ Composite', 'Dow Jones'])]
    other_indices_data = data[~data['Index'].isin(['NASDAQ Composite', 'Dow Jones'])]

    #plotting the Nasdaq and Dow Jones
    fig1, ax1 = plt.subplots(figsize = (20, 10))
    for index in ixic_dji_data['Index'].unique():
        index_data = ixic_dji_data[ixic_dji_data['Index'] == index]
        ax1.plot(index_data['Date'], index_data['Close'], label = index)

    ax1.set_xlabel('Year', fontsize = 24)
    ax1.set_ylabel('Closing Price', fontsize = 24)
    ax1.legend(loc = 'upper left', prop = {'size': 24})

    plt.yticks(fontsize = 24)
    plt.xticks(fontsize = 24)

    #plotting the S&P, RUT and FTSE
    fig2, ax2 = plt.subplots(figsize = (20, 10))
    for index in other_indices_data['Index'].unique():
        index_data = other_indices_data[other_indices_data['Index'] == index]
        ax2.plot(index_data['Date'], index_data['Close'], label = index)

    ax2.set_xlabel('Year', fontsize = 24)
    ax2.set_ylabel('Closing Price', fontsize = 24)
    ax2.legend(loc = 'upper left', prop = {'size': 24})

    plt.yticks(fontsize = 24)
    plt.xticks(fontsize = 24)
    plt.show()

indices = ["^GSPC", "^RUT", "^IXIC", "^FTSE", "^DJI"]
dfs = []

for index in indices:
    #retrieving the data from Yahoo Finance
    data = yf.download(index, start = '2010-01-01', end = '2024-01-01')
    data['Index'] = index
    data = data.reset_index()
    dfs.append(data)

data = pd.concat(dfs, ignore_index = True, sort = False)[['Date', 'Close', 'Index']]
plot_index(data)
