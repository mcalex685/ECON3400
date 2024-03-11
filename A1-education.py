# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 19:36:14 2024

@author: alexm
"""
#data for 20-25 year olds 
import pandas as pd
import matplotlib.pyplot as plt
from fredapi import Fred
fred_key = 'd95e779ea902dea8c3a83e5ee0f4f740'

#1. Create the Fred object
fred=Fred(api_key=fred_key)

#Employment Level - Bachelor's Degree and Higher, 25 Yrs. & over
ELB=fred.get_series('LNS12027662')
# Convert Series to DataFrame
ELB_df = ELB.to_frame()
ELB_df.columns=['EL Bachelors']
ELB_df.index=ELB_df.index.date
#plt.plot(ELB_df.index, ELB_df['EL Bachelors'], linestyle='-')
# Convert index to datetime index
ELB_df.index = pd.to_datetime(ELB_df.index)
#annual averages 
an_averages_ELB= ELB_df.resample('Y').mean()

#Employment Level - High School Graduates, No College, 25 Yrs. & over
ELH=fred.get_series('LNS12027660')
# Convert Series to DataFrame
ELH_df = ELH.to_frame()
ELH_df.columns=['EL HighSchool']
ELH_df.index=ELH_df.index.date
#plt.plot(ELH_df.index, ELH_df['EL HighSchool'], linestyle='-')
# Convert index to datetime index
ELH_df.index = pd.to_datetime(ELH_df.index)
#annual averages 
an_averages_ELH= ELH_df.resample('Y').mean()

#Working Age Population: Aged 15-24: All Persons for United States
WAP=fred.get_series('LFWA24TTUSM647S')
# Convert Series to DataFrame
WAP_df= WAP.to_frame()
WAP_df.columns=['Working Age Population: Aged 15-24']
WAP_df.index=WAP_df.index.date
#resacle to thousands 
WAP_df[['Working Age Population: Aged 15-24']]/=1000
#plt.plot(WAP_df.index, WAP_df['Working Age Population: Aged 15-24'], linestyle='-')


# Convert index to datetime index
WAP_df.index = pd.to_datetime(WAP_df.index)
#annual averages 
an_averages_WAP= WAP_df.resample('Y').mean()

#join average LFP plots 
avgAll_df=an_averages_ELB.join([an_averages_ELH,an_averages_WAP])
avgAll_df=avgAll_df.dropna()

# Calculate year-to-year growth rates
avgAll_df['B Growth']= avgAll_df['EL Bachelors'].pct_change() * 100
avgAll_df['HS Growth']= avgAll_df['EL HighSchool'].pct_change() * 100
# Calculate column averages
column_averages = avgAll_df.mean()

#plot
plt.plot(avgAll_df.index, avgAll_df['EL Bachelors'], color='blue')
plt.plot(avgAll_df.index, avgAll_df['EL HighSchool'], linestyle='-', color='red')
#plt.plot(avgAll_df.index, avgAll_df['Working Age Population: Aged 15-24'], linestyle='-', color='Green')
plt.title('Average Annual Employment Level of Individuals in the US 25\n years & over 1992-2024')
plt.xlabel('Year')
plt.ylabel('# of persons (thousands)')
#plt.ylim(0, 100)  
plt.legend(labels=['Bachelors Degree and Higher','High School Graduates, No College'])

#LFP 15-24 yrs in percent
LFP=fred.get_series('LRAC24TTUSM156S')
# Convert Series to DataFrame
LFP_df = LFP.to_frame()
LFP_df.columns=['LFP']
LFP_df.index=LFP_df.index.date
#plt.plot(LFP_df.index, LFP_df['LFP'], linestyle='-')

#annual averages 
# Convert index to datetime index
LFP_df.index = pd.to_datetime(LFP_df.index)
annual_averages_LFP= LFP_df.resample('Y').mean()
#plt.plot(annual_averages_LFP.index, annual_averages_LFP['LFP'], linestyle='-')



import numpy as np
import statsmodels.api as sm
# Convert dates to ordinal numbers
avgAll_df['ordinal_date'] = avgAll_df.index.to_series().apply(lambda x: x.toordinal())

# Create the model
X = avgAll_df['ordinal_date']  # Independent variable
y = avgAll_df['EL Bachelors']  # Dependent variable

# Add a constant to the independent variable (required for statsmodels)
X = sm.add_constant(X)

# Fit the model bachlors 
model = sm.OLS(y, X).fit()

# Predict values for new dates
new_dates = pd.date_range(start='1960-01-01', end='2024-01-01', freq='Y')
new_ordinal_dates = new_dates.to_series().apply(lambda x: x.toordinal())
new_X = sm.add_constant(new_ordinal_dates)
predicted_values = model.predict(new_X)

# Calculate prediction intervals
from statsmodels.sandbox.regression.predstd import wls_prediction_std
pred_std, lower, upper = wls_prediction_std(model, exog=new_X, alpha=0.05)

# Create the model highschool 
X2 = avgAll_df['ordinal_date']  # Independent variable
y2 = avgAll_df['EL HighSchool']  # Dependent variable

# Add a constant to the independent variable (required for statsmodels)
X2 = sm.add_constant(X2)

# Fit the model
model2 = sm.OLS(y2, X2).fit()

# Predict values for new dates
new_dates = pd.date_range(start='1960-01-01', end='2024-01-01', freq='Y')
new_ordinal_dates = new_dates.to_series().apply(lambda x: x.toordinal())
new_X2 = sm.add_constant(new_ordinal_dates)
predicted_values2 = model2.predict(new_X2)

# Calculate prediction intervals
from statsmodels.sandbox.regression.predstd import wls_prediction_std
pred_std2, lower2, upper2 = wls_prediction_std(model2, exog=new_X2, alpha=0.05)



# Plot the original data and the regression line
#plt.plot(avgAll_df.index, y, label='Data-Bachelors Degree')
#plt.plot(avgAll_df.index, y2, label='Data-High School', color='orange')
plt.plot(annual_averages_LFP.index, annual_averages_LFP['LFP'], linestyle='-', color = 'Blue')
# Plot the predicted values
plt.plot(new_dates, predicted_values, color='green', label='Predicted Values-Bachelors Degree')
plt.plot(new_dates, predicted_values2, color='red', label='Predicted Values-High School')
#prediction interval
plt.fill_between(new_dates, lower, upper, color='green', alpha=0.2, label='95% Prediction Interval')
plt.fill_between(new_dates, lower2, upper2, color='red', alpha=0.2, label='95% Prediction Interval')
plt.xlabel('Year')
plt.ylabel('# of persons (thousands)')
plt.title('Average Annual Employment Level of Individuals in the US\n 25 years-old & over')
plt.ylim(0, 60000) 
plt.legend(labels= ['Data-Bachelors Degree','Data-High School','Predicted Values-Bachelors Degree','Predicted Values-High School'])
plt.show()

#boxplots 
#Extrapolated data 
# Define the range of dates
start_date = '1970-01-01'
end_date = '1980-01-10'
#ba
# Filter the DataFrame to include only the rows within the specified date range
BA70 = predicted_values[(predicted_values.index >= start_date) & (predicted_values.index <= end_date)]
BA70_df = BA70.to_frame()
# Calculate the mean of a specific column (e.g., 'column_name')
BA70_mean = BA70_df[0].mean()
#hs
# Filter the DataFrame to include only the rows within the specified date range
HS70 = predicted_values2[(predicted_values2.index >= start_date) & (predicted_values2.index <= end_date)]
HS70_df = HS70.to_frame()
# Calculate the mean of a specific column (e.g., 'column_name')
HS70_mean = HS70_df[0].mean()

#observed data 
# Define the range of dates
start_date = '2010-01-01'
end_date = '2020-01-10'
#ba
# Filter the DataFrame to include only the rows within the specified date range
BA10 = an_averages_ELB[(an_averages_ELB.index >= start_date) & (an_averages_ELB.index <= end_date)]
# Calculate the mean of a specific column (e.g., 'column_name')
BA10_mean = BA10['EL Bachelors'].mean()
#hs
# Filter the DataFrame to include only the rows within the specified date range
HS10 = an_averages_ELH[(an_averages_ELH.index >= start_date) & (an_averages_ELH.index <= end_date)]
# Calculate the mean of a specific column (e.g., 'column_name')
HS10_mean = HS10['EL HighSchool'].mean()

# Create a figure for the box plots
plt.figure(figsize=(10, 6))

# List of DataFrames you want to plot
dataframes_to_plot = [BA70_df, BA10, HS70_df, HS10]

# Plot box plots for each DataFrame
for i, df in enumerate(dataframes_to_plot):
    plt.subplot(1, len(dataframes_to_plot), i+1)  # Create subplots
    df.boxplot(color='C{}'.format(i))  # Use the color cycle for different colors
    df.boxplot()
    #plt.title(f'{i+1}')  # Add title
    #plt.xlabel('Variables')  # Add x-axis label
    plt.ylabel('# of persons (thousands)')     # Add y-axis label
    plt.xticks(rotation=45)  # Rotate x-axis labels if needed
    plt.grid(False)          # Disable grid lines if not needed

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()

#graph predicted values and LFP 
# Plot data on separate axes
fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('Year')
ax1.set_ylabel('Labour Force Participation Rate (%)', color=color)
ax1.plot(annual_averages_LFP.index, annual_averages_LFP['LFP'], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
#color = 'tab:blue'
ax2.set_ylabel('# of persons (thousands)')
ax2.plot(new_dates, predicted_values, color='green', label='Predicted Values-Bachelors Degree')
ax2.plot(new_dates, predicted_values2, color='red', label='Predicted Values-High School')
#prediction interval
plt.fill_between(new_dates, lower, upper, color='green', alpha=0.2, label='95% Prediction Interval')
plt.fill_between(new_dates, lower2, upper2, color='red', alpha=0.2, label='95% Prediction Interval')
ax2.tick_params(axis='y')


plt.title('Labour FOrce Paticpation Rate (15-24) and Predicted Employment\n Level for 1960-2024')
plt.legend(labels= ['Bachelors Degree','High School'])
plt.ylim(0, 60000) 
fig.tight_layout()
plt.show()

#graph given values and LFP since 1992 
# Specify the index date
given_date = pd.to_datetime('1992-01-01')

# Filter the DataFrame to include only the rows with index dates greater than the specified date
filtered_df = annual_averages_LFP[annual_averages_LFP.index > given_date]

# Plot data on separate axes
fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('Year')
ax1.set_ylabel('Labour Force Participation Rate (%)', color=color)
ax1.plot(filtered_df.index, filtered_df['LFP'], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
#color = 'tab:blue'
ax2.plot(avgAll_df.index, y, label='Data-Bachelors Degree', color='blue')
ax2.plot(avgAll_df.index, y2, label='Data-High School', color='red')
ax2.set_ylabel('# of persons (thousands)')
ax2.tick_params(axis='y')


plt.title('Labour Force Paticpation Rate (age 15-24) and Employment\n Level (age 25+) for 1992-2024')
plt.legend(labels= ['Bachelors Degree','High School'], loc ='lower left')
plt.ylim(0, 60000) 
fig.tight_layout()
plt.show()

#find correlation year to year 
cor_df=an_averages_ELB.join([an_averages_ELH,annual_averages_LFP])
cor_df.columns=['Bachelors','Highschool', 'Labour Force Partipation']
cor_matrix=cor_df.corr()
# Plot the correlation matrix as a heatmap
import seaborn as sns
plt.figure(figsize=(8, 6))
sns.heatmap(cor_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix')
plt.show()

