# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 16:07:12 2024

@author: alexm
"""
import pandas as pd
import matplotlib.pyplot as plt
from fredapi import Fred
fred_key = 'd95e779ea902dea8c3a83e5ee0f4f740'

#1. Create the Fred object
fred=Fred(api_key=fred_key)

#2. Search for economic data
#popularity gives what mmost people ask for on a given search 
sp_search = fred.search('S&P', order_by='popularity')
# 3. Pull raw data and plot, T10Y2Y, CPIAUCSL, FEDFUNDS, JTSTSR, CIVPART
SP500 = fred.get_series(series_id='CSUSHPINSA')
# Convert Series to DataFrame
sp_df = SP500.to_frame()
sp_df=sp_df.dropna()
sp_df.columns=['S&P500']
sp_df.index=sp_df.index.date

# 4. UNEMP RATE
unemp_results=fred.search('unemployment')
unrate=fred.get_series('UNRATE')
# Convert Series to DataFrame
unrate_df = unrate.to_frame()
unrate_df.columns=['unemployment']
unrate_df.index=unrate_df.index.date

#GET LFP in percent
LFP=fred.get_series('CIVPART')
# Convert Series to DataFrame
LFP_df = LFP.to_frame()
LFP_df.columns=['LFP']
LFP_df.index=LFP_df.index.date

joined_df=sp_df.join([unrate_df,LFP_df])

# Plot data on separate axes
fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Year')
ax1.set_ylabel('S&P500', color=color)
ax1.plot(joined_df.index, joined_df['S&P500'], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Unemployment Rate(%)', color=color)
ax2.plot(joined_df.index,joined_df['unemployment'], color=color)
ax2.tick_params(axis='y', labelcolor=color)

ax3 = ax1.twinx()
color = 'tab:green'
ax3.spines['right'].set_position(('outward', 60))
ax3.set_ylabel('Labor Force Participation Rate(%)', color=color)
ax3.plot(joined_df.index, joined_df['LFP'], color=color)
ax3.tick_params(axis='y', labelcolor=color)

plt.title('Plot of Key Economic Indicators from 1987 to 2024')
fig.tight_layout()
plt.show()

#Average LFP 2012-2020 
# Define the date range
start_date = pd.to_datetime('2012-01-01').date()
end_date = pd.to_datetime('2020-01-01').date()

# Slice the DataFrame for the specified date range and calculate the mean
average_value = LFP_df.loc[start_date:end_date, 'LFP'].mean()

# Create a correlation matrix
import seaborn as sns
correlation_matrix = joined_df.corr()

# Plot the correlation matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix')
plt.show()