"""
This script plots the percentage change per month in bearish sentiment and S&P 500 returns.
"""
import pandas as pd
import os
import matplotlib.pyplot as plt



file_path = r'datasets\retrieved_data'
contrarian = pd.read_csv(os.path.join(file_path, "contrarian.csv"))
contrarian['Date'] = pd.to_datetime(contrarian['Date'], format = 'mixed')


plt.figure(figsize=(20, 10))
plt.plot(contrarian['Date'], contrarian['Bearish Percent'], label = 'Bearish', linestyle = '-', marker = 'o', markersize = 6, color = 'black', linewidth = 2)
plt.plot(contrarian['Date'], contrarian['Returns'], label = 'Returns', linestyle = '--', marker = 's', markersize = 6, color = 'gray', linewidth = 2)

contrarian['Year'] = contrarian['Date'].dt.year
yearly_ticks = contrarian.groupby('Year')['Date'].first()

plt.xticks(ticks = yearly_ticks, labels = yearly_ticks.dt.year, fontsize = 24)
plt.xlabel('Year', fontsize = 24)
plt.ylabel('Percentual Change', fontsize = 24)
plt.yticks(fontsize = 24)
plt.legend(fontsize = 24, frameon = True, framealpha = 1, shadow = True, borderpad = 1)
plt.grid(True, which = 'both', linestyle = '--', linewidth = 0.5)
plt.tight_layout()
plt.show()


