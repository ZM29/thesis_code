"""
This script plots the ICS and AAII sentiment scores.
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def plot_ics(ics):

    ics.set_index('Date', inplace = True)

    plt.figure(figsize = (20, 10))
    plt.plot(ics.index, ics['ICS_ALL'], linestyle = '-', marker = 'o', markersize = 4)
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xlabel('Year', fontsize = 24)
    plt.ylabel('Index Value (1966=100)', fontsize = 24)
    plt.grid(True, which='both', linestyle='--', linewidth = 0.5)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.tight_layout()
    plt.show()

def plot_aaii(weekly_data):

    #resampling to monthly and taking the difference
    monthly_data = weekly_data.set_index('Date').resample('M').mean().reset_index()
    monthly_data['Sentiment_Difference'] = monthly_data['Bullish'] - monthly_data['Bearish']

    plt.figure(figsize = (20, 10))
    plt.plot(monthly_data['Date'], monthly_data['Sentiment_Difference'], label = 'Sentiment Difference', linewidth = 2, marker = 'o', color = 'b')

    monthly_data['Year'] = monthly_data['Date'].dt.year
    yearly_ticks = monthly_data.groupby('Year')['Date'].first()

    plt.xticks(ticks = yearly_ticks, labels = yearly_ticks.dt.year, fontsize = 24)
    plt.xlabel('Year', fontsize = 24)
    plt.ylabel('Sentiment Difference (Bullish - Bearish)', fontsize = 24)
    plt.yticks(fontsize = 24)
    plt.legend(fontsize = 24)
    plt.grid(True, which = 'both', linestyle = '--', linewidth = 0.5)
    plt.tight_layout()
    plt.show()


file_path = r'datasets\retrieved_data'
output_path = r'datasets\retrieved_data'
    
ics = pd.read_csv(os.path.join(file_path, "csi.csv"))
aaii = pd.read_csv(os.path.join(file_path, "investor_index.csv"))

aaii['Date'] = pd.to_datetime(aaii['Date'], format= 'mixed')
ics['Date'] = pd.to_datetime(ics['Date'], format= 'mixed')

#filtering the data
ics = ics[ics['Date'] > '30-12-2009']
aaii = aaii[aaii['Date'] > '1-1-2010']
aaii = aaii[['Date','Bullish','Bearish']]
    
plot_ics(ics)
plot_aaii(aaii)
