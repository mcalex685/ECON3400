# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 13:21:43 2024

@author: alexm
"""
#data for 20-25 year olds 
import pandas as pd
import matplotlib.pyplot as plt
from fredapi import Fred
fred_key = 'd95e779ea902dea8c3a83e5ee0f4f740'

#1. Create the Fred object
fred=Fred(api_key=fred_key)
#S&P500
SP500 = fred.get_series(series_id='CSUSHPINSA')
# Convert Series to DataFrame
sp_df = SP500.to_frame()
sp_df=sp_df.dropna()
sp_df.columns=['S&P500']
sp_df.index=sp_df.index.date

#UNEMP RATE
unrate=fred.get_series('LNS14024887')
# Convert Series to DataFrame
unrate_df = unrate.to_frame()
unrate_df.columns=['unemployment']
unrate_df.index=unrate_df.index.date

#LFP in percent
LFP=fred.get_series('LNS11300036')
# Convert Series to DataFrame
LFP_df = LFP.to_frame()
LFP_df.columns=['LFP']
LFP_df.index=LFP_df.index.date

joined_df=sp_df.join([unrate_df,LFP_df])

# Create a correlation matrix
import seaborn as sns
correlation_matrix = joined_df.corr()

# Plot the correlation matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix')
plt.show()

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

plt.title('Plot of Key Economic Indicators for Young People from 1987 to 2024')
fig.tight_layout()
plt.show()

#histogram of labour force partipation 
#annual averages 
# Convert index to datetime index
LFP_df.index = pd.to_datetime(LFP_df.index)
annual_averages = LFP_df.resample('Y').mean()

# Create a colormap
cmap = plt.cm.get_cmap('viridis')

# Map years to values between 0 and 1
years = annual_averages.index.year
min_year = years.min()
max_year = years.max()
scaled_years = (years - min_year) / (max_year - min_year)

# Plot a bar graph with dates on the x-axis and colors from the colormap
plt.bar(annual_averages.index, annual_averages['LFP'], color=cmap(scaled_years), width = 375, alpha=1,label='Annual Averages')
#plt.plot(sp_df.index, sp_df['S&P500'], color='red', label='S&P 500')  # Add S&P 500 line plot
plt.title('Average Annual Labour Force Participation Rate of 20-24 year-olds from 1948-2024')
plt.xlabel('Year')
plt.ylabel('Labour Force Participation Rate(%)')
plt.xticks(rotation=45)
plt.grid(axis='y')
# Adjust the y-axis range
plt.ylim(55, 90)  # Set the range from 0 to 60

# Fit a linear regression model
import numpy as np
coefficients = np.polyfit(range(len(annual_averages)), annual_averages['LFP'], 1)
trend_line = np.polyval(coefficients, range(len(annual_averages)))

# Plot the trend line
plt.plot(annual_averages.index, trend_line, color='black', linestyle='-')

# Add legend
#plt.legend()

# Show the plot
plt.tight_layout()
plt.show()

#decade averages 
# Specify the starting year for calculating decade averages
starting_year = 1950

# Filter the DataFrame to include data from the starting year onwards
lessLFP_df = LFP_df[LFP_df.index.year >= starting_year]

