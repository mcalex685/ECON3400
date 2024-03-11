# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 15:59:39 2024

@author: alexm
"""
#demographics 
import pandas as pd
import matplotlib.pyplot as plt
from fredapi import Fred
fred_key = 'd95e779ea902dea8c3a83e5ee0f4f740'

#1. Create the Fred object
fred=Fred(api_key=fred_key)

#LFP 15-24 yrs in percent
LFP=fred.get_series('LRAC24TTUSM156S')
# Convert Series to DataFrame
LFP_df = LFP.to_frame()
LFP_df.columns=['LFP']
LFP_df.index=LFP_df.index.date
plt.plot(LFP_df.index, LFP_df['LFP'], linestyle='-')

#annual averages 
# Convert index to datetime index
LFP_df.index = pd.to_datetime(LFP_df.index)
annual_averages_LFP= LFP_df.resample('Y').mean()
plt.plot(annual_averages_LFP.index, annual_averages_LFP['LFP'], linestyle='-')

#LFP 15-24 yrs FEMALE in percent
LFPfem=fred.get_series('LRAC24FEUSM156S')
# Convert Series to DataFrame
LFPfem_df = LFPfem.to_frame()
LFPfem_df=LFPfem_df.dropna()
LFPfem_df.columns=['LFPfem']
LFPfem_df.index=LFPfem_df.index.date
plt.plot(LFPfem_df.index, LFPfem_df['LFPfem'], linestyle='-')

#annual averages 
# Convert index to datetime index
LFPfem_df.index = pd.to_datetime(LFPfem_df.index)
#averages 
annual_averages_LFPfem= LFPfem_df.resample('Y').mean()

#LFP 15-24 yrs MALE in percent
LFPmal=fred.get_series('LRAC24MAUSM156S')
# Convert Series to DataFrame
LFPmal_df = LFPmal.to_frame()
LFPmal_df=LFPmal_df.dropna()
LFPmal_df.columns=['LFPmal']
LFPmal_df.index=LFPmal_df.index.date
plt.plot(LFPmal_df.index, LFPmal_df['LFPmal'], linestyle='-')

#annual averages 
# Convert index to datetime index
LFPmal_df.index = pd.to_datetime(LFPmal_df.index)
#averages 
annual_averages_LFPmal= LFPmal_df.resample('Y').mean()

#join average LFP plots 
avgLFP_df=annual_averages_LFP.join([annual_averages_LFPfem,annual_averages_LFPmal])
avgLFP_df=avgLFP_df.dropna()
# Calculate year-to-year growth rates
avgLFP_df['Growth Rate Agg'] = avgLFP_df['LFP'].pct_change() * 100
avgLFP_df['Growth Rate Fem'] = avgLFP_df['LFPfem'].pct_change() * 100
avgLFP_df['Growth Rate Mal'] = avgLFP_df['LFPmal'].pct_change() * 100
#decade averages 
avgLFP10_df=avgLFP_df.resample('10Y').mean()
avgLFP10_df['diff']=avgLFP10_df['LFPmal']-avgLFP10_df['LFPfem']
avgLFP10_df=avgLFP10_df.drop(index = '2030-12-31')

# Plotting the heatmap of decade differences 
import seaborn as sns
plt.figure(figsize=(8, 6))
heatmap = sns.heatmap(avgLFP10_df[['diff']], annot=True, cmap='coolwarm', fmt=".2f", cbar=True)
plt.title('Heatmap of Differences in Labour Force Participation Rate\n of between 15-24 year-old Males and Females')
plt.xlabel('')
plt.ylabel('Decade')
xlab = ['Difference (%)']
ylab = ['1960s','1970s','1980s','1990s','2000s','2010s', '2020s'] 
heatmap.set_yticklabels(ylab)  # Custom labels for y-axis
heatmap.set_xticklabels(xlab) 
plt.show()

# Plotting the heatmap of decade differences 
plt.figure(figsize=(8, 6))
heatmap = sns.heatmap(avgLFP_df[['Growth Rate Fem','Growth Rate Mal']], annot=False, cmap='coolwarm', fmt=".2f", cbar=True)
plt.title('Heatmap of Growth in Labour Force Participation Rate\n of between 15-24 year-old Males and Females')
plt.xlabel('Growth Rate (%)')
plt.ylabel('Year')
xlab = ['Female','Male']
ylab = list(range(1960, 2024,3)) 
heatmap.set_yticklabels(ylab)  # Custom labels for y-axis
heatmap.set_xticklabels(xlab) 
plt.show()


#plot
plt.plot(avgLFP_df.index, avgLFP_df['LFP'], color='blue')
plt.plot(avgLFP_df.index, avgLFP_df['LFPfem'], linestyle='-', color='red')
plt.plot(avgLFP_df.index, avgLFP_df['LFPmal'], linestyle='-', color='Yellow')
plt.title('Average Annual Labour Force Participation Rate of 15-24\n year-olds from 1960-2024')
plt.xlabel('Year')
plt.ylabel('Labour Force Participation Rate(%)')
#plt.ylim(0, 100)  
plt.legend(labels=['Aggregate','Females', 'Males'])

#bar graph of rates 
plt.bar(avgLFP_df.index, avgLFP_df['Growth Rate Fem'], color='Red', width = 350, alpha=0.5,label='Annual Averages')
plt.bar(avgLFP_df.index, avgLFP_df['Growth Rate Mal'], color='Yellow', width = 350, alpha=0.5,label='Annual Averages')
plt.title('Average Annual Labour Force Participation Growth Rate of\n 15-24 year-olds from 1960-2024')
plt.xlabel('Year')
plt.ylabel('Growth Rate(%)') 
plt.legend(labels=['Females', 'Males'])


#Working Age Population: Aged 15-24: All Persons for United States
WAP=fred.get_series('LFWA24TTUSM647S')
# Convert Series to DataFrame
WAP_df= WAP.to_frame()
WAP_df.columns=['Working Age Population: Aged 15-24']
WAP_df.index=WAP_df.index.date
plt.plot(WAP_df.index, WAP_df['Working Age Population: Aged 15-24'], linestyle='-')



