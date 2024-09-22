#!/usr/bin/env python
# coding: utf-8

# In[43]:


get_ipython().run_line_magic('matplotlib', 'inline')
# Mapping
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.dates as dates
from matplotlib.dates import DateFormatter
# Stats
import pymannkendall as mk
# Data
import netCDF4 as nc
import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')


# In[44]:


pwd # Confirming your current path


# In[45]:


# Replace my path with your path to where your data lives
DATA_PATH = os.path.join("/path/to/data")
INFILE = "data.nc" # replace with your filename if named differently


# In[46]:


nc_ = nc.Dataset(os.path.join(DATA_PATH, INFILE))
nc_


# In[47]:


nc_.variables['analysed_sst'].dimensions
#nc_.variables['zos'].dimensions
#nc_.variables['chlor_a'].dimensions


# In[48]:


# Pull out your SST / SSH and co-variables 
# (this step may take time if your dataset is big)
#var = nc_.variables['zos'] #SSH variable
#nc_.variables['chlor_a'].dimensions #chlorophyll-a variable
var = nc_.variables['analysed_sst'][:]-273.15  # [:,0,:,:] if you need to remove depth dimension, else [:]
time= nc_.variables['time']
lat = nc_.variables['latitude'][:]
lon = nc_.variables['longitude'][:]
print('Number of timesteps, lat, and lon =',var.shape)


# In[49]:


# Get your dates into a user-friendly format: dtime is your new date variable
caldr = time.getncattr('calendar')
units = time.getncattr('units')
dtime = nc.num2date(time[:], units, calendar = caldr, only_use_cftime_datetimes = False)


# In[50]:


#define area of CBG within data 
LAT_ss = [-23.6, -23.08]
LON_ss = [151.36, 152.36]
lat_min_index = np.argmin(np.abs(lat - LAT_ss[0]))
lat_max_index = np.argmin(np.abs(lat - LAT_ss[1]))
lon_min_index = np.argmin(np.abs(lon - LON_ss[0]))
lon_max_index = np.argmin(np.abs(lon - LON_ss[1]))


# In[51]:


#create spatial subset of reef area 
var_ss = []
timesteps = var.shape[0]
for x in range(timesteps):
    subset = np.nanmean(var[x,lat_min_index:lat_max_index,lon_min_index:lon_max_index])
    var_ss.append(subset)


# In[52]:


# Polyfit for linear regression (y=m*x+c)
x_date = dtime
x_num1 = dates.date2num(x_date)
m1,c1,*_= np.polyfit(x_num1, var_ss, 1, full=True)
fit1 = np.poly1d(m1)

# Change over time
up1 = (m1[0] * len(var_ss))
print('My variable (SST/ SSH) appears to have changed by', ("%.3f" % up1), 
      'degrees / meters over the full timeseries in this spot.')


# In[53]:


# Change per decade
days_per_decade = 365.25 * 10  # Average days in a decade considering leap years
change_per_decade = m1[0] * days_per_decade
print('My variable (SST/SSH) changes by', ("%.3f" % change_per_decade), 
      'degrees/meters per decade in this spot.')


# In[54]:


# Create a DataFrame with the date range and var_ss values
df = pd.DataFrame({'Date': dtime, 'Var': var_ss})

# Group by month and filter for min and max values
var_ss_min = df.loc[df.groupby(df['Date'].dt.to_period('Y'))['Var'].idxmin()] #change Y to M for monthly min variable
var_ss_max = df.loc[df.groupby(df['Date'].dt.to_period('Y'))['Var'].idxmax()] #change Y to M for monthly max variable


# In[55]:


def polyfit_linear_regression(dataframe):
    x_date = dataframe['Date']
    x_num = dates.date2num(x_date)
    m, c, *_ = np.polyfit(x_num, dataframe['Var'], 1, full=True)
    fit = np.poly1d(m)
    change_over_time = m[0] * len(dataframe)
    return m, c, fit, change_over_time

# Apply to min DataFrame
m_min, c_min, fit_min, change_min = polyfit_linear_regression(var_ss_min)
print('The minimum values appear to have changed by', ("%.3f" % change_min), 'degrees/meters over the full timeseries.')

# Apply to max DataFrame
m_max, c_max, fit_max, change_max = polyfit_linear_regression(var_ss_max)
print('The maximum values appear to have changed by', ("%.3f" % change_max), 'degrees/meters over the full timeseries.')



# In[56]:


# Check significance of trends using Mann-Kendall test

trend1 = mk.original_test(var_ss)
print("Mean trend is",trend1[0], ", P-value =", ("%.10f" % trend1[2]), 
      ", Slope =", ("%.6f" % trend1[5]), ", Intercept =", ("%.2f" % trend1[8]))

trend_min = mk.original_test(var_ss_min['Var'])
print("Trend in minimum values is", trend_min[0], ", P-value =", ("%.10f" % trend_min[2]), 
      ", Slope =", ("%.6f" % trend_min.slope))

trend_max = mk.original_test(var_ss_max['Var'])
print("Trend in maximum values is", trend_max[0], ", P-value =", ("%.10f" % trend_max[2]), 
      ", Slope =", ("%.6f" % trend_max.slope))


# In[57]:


# Calculating median, max and min SST values, and determining where they sit in the timeseries using index
max = var_ss.index(np.nanmax(var_ss))
min = var_ss.index(np.nanmin(var_ss))
M = (np.nanmean(var_ss))
D = dtime[max]

print ('My Pixel:','mean is', ("%.2f" % M), 
       'with ' 'max [',("%.2f" % np.nanmax(var_ss)), 'at index', max,']',  
       '& ' 'min [',("%.2f" % np.nanmin(var_ss)), 'at index', min,']')

df_ss = pd.DataFrame({'Var':var_ss, 'Time':dtime})
peak1= df_ss.iloc[df_ss['Var'].idxmax()]
print('The date of the highest value reached for my specific point of interest was:',peak1[1])


# In[58]:


import numpy as np
import pandas as pd
import matplotlib.dates as dates
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, YearLocator

# Assuming var_ss_min and var_ss_max are already defined
# Here var_ss_min and var_ss_max contain the filtered minimum and maximum values per year respectively

def polyfit_linear_regression(dataframe):
    x_date = dataframe['Date']
    x_num = dates.date2num(x_date)
    m, c, *_ = np.polyfit(x_num, dataframe['Var'], 1, full=True)
    fit = np.poly1d(m)
    change_over_time = m[0] * len(dataframe)
    return m, c, fit, change_over_time

# Apply to min DataFrame
m_min, c_min, fit_min, change_min = polyfit_linear_regression(var_ss_min)
print('The minimum values appear to have changed by', ("%.3f" % change_min), 'degrees/meters over the full timeseries.')

# Apply to max DataFrame
m_max, c_max, fit_max, change_max = polyfit_linear_regression(var_ss_max)
print('The maximum values appear to have changed by', ("%.3f" % change_max), 'degrees/meters over the full timeseries.')

# Create the main plot with the original data
fig1, ax = plt.subplots(figsize=(10, 6), dpi=250)

# Make sure the datetime is properly formatted as pandas datetime if not already
if not pd.api.types.is_datetime64_any_dtype(dtime):
    dtime = pd.to_datetime(dtime)  # Convert to datetime if not already in datetime format

# Set x-axis to display every year
ax.xaxis.set_major_locator(YearLocator(5))  # Ticks at every 5 years
ax.xaxis.set_major_formatter(DateFormatter('%Y'))  # Format ticks to display the year

plt.setp(ax.get_xticklabels(), rotation=0)  # Rotate for better readability

# Plot the main data
plt.plot(dtime, var_ss, linewidth=0.1, marker='o', markersize=1, c='navy')
plt.plot(dtime, var_ss, linewidth=0.7, c='silver', linestyle='-')

# Add trendline to plot
x_fit = np.linspace(dates.date2num(dtime).min(), dates.date2num(dtime).max())
plt.plot(dates.num2date(x_fit), fit1(x_fit), color='navy', linewidth=1, label='Original Trendline')

# Add min regression line to plot
x_fit_min = np.linspace(dates.date2num(var_ss_min['Date']).min(), dates.date2num(var_ss_min['Date']).max())
plt.plot(dates.num2date(x_fit_min), fit_min(x_fit_min), "b-", color='navy',linestyle='--', alpha=0.5, linewidth=0.7, label='Min Trendline')

# Add max regression line to plot
x_fit_max = np.linspace(dates.date2num(var_ss_max['Date']).min(), dates.date2num(var_ss_max['Date']).max())
plt.plot(dates.num2date(x_fit_max), fit_max(x_fit_max), "g-",color='navy', linestyle='--', linewidth=0.7,alpha=0.5,label='Max Trendline')

# Limit x-axis to the range of the data
ax.set_xlim([dtime.min(), dtime.max()])

# Add vertical lines for specific years
highlight_years = [2016, 2017, 2020]
for year in highlight_years:
    ax.axvline(pd.Timestamp(f'{year}-01-01'), color='#CD5C5C', linestyle='-', linewidth=0.9)

# Add labels
plt.xlabel('')
plt.ylabel('Sea Surface Temperature (°C)')
#plt.legend()
ax.grid(False)

# Display the plot
plt.tight_layout()
plt.savefig('trendlineSSH.png', dpi=300, bbox_inches='tight')
plt.show()


# In[59]:


# Create a DataFrame with the date range and var_ss values
df = pd.DataFrame({'Date': dtime, 'Var': var_ss})


# In[60]:


import seaborn as sns

# Extract the year from the 'Dates' column
df['Year'] = df['Date'].dt.year

# Set the style for the plot
sns.set(style="whitegrid")

# Define the years to be highlighted in red
highlight_years = [1998, 2002, 2016, 2017, 2020, 2022]

# Create a color palette with red for the highlight years and default for others
palette = {year: 'red' if year in highlight_years else 'C0' for year in df['Year'].unique()}

# Create the box plot
plt.figure(figsize=(14, 7))
sns.boxplot(x='Year', y='Var', data=df, palette=palette, order=sorted(df['Year'].unique()))

# Enhance the labels and title
plt.title('Box Plot of VAR by Year', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('VAR', fontsize=14)
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability

# Show the grid for better readability
plt.grid(True, linestyle='--', alpha=0.7)

# Display the plot
plt.show()


# In[61]:


import matplotlib.patches as mpatches
# Set the style for the plot
sns.set(style="whitegrid")

# El Niño and La Niña years
el_nino_years = [1972, 1973, 1978, 1980, 1983, 1987, 1992, 1995, 1998, 2003, 2007, 2010, 2016, 2023]
la_nina_years = [1971, 1974, 1976, 1989, 1999, 2000, 2008, 2011, 2012, 2021, 2022]

# Create a color palette with red for El Niño years, green for La Niña years, and default for others
palette = {year: 'red' if year in el_nino_years else 'green' if year in la_nina_years else 'C0' for year in df['Year'].unique()}

# Create the box plot
plt.figure(figsize=(14, 7))
sns.boxplot(x='Year', y='Var', data=df, palette=palette, order=sorted(df['Year'].unique()))

# Enhance the labels and title
plt.title('Box Plot of SST ENSO highlights', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('VAR', fontsize=14)
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability

# Add a legend
red_patch = mpatches.Patch(color='red', label='El Niño Year')
green_patch = mpatches.Patch(color='green', label='La Niña Year')
default_patch = mpatches.Patch(color='C0', label='Neutral Year')
plt.legend(handles=[red_patch, green_patch, default_patch], loc='lower right')

# Show the grid for better readability
plt.grid(True, linestyle='--', alpha=0.7)

plt.savefig('boxplotsENSOSSH.png', dpi=300, bbox_inches='tight')

# Display the plot
plt.show()


# In[62]:


# Filter the DataFrame to include only years from 2010 to 2022
filtered_df = df[(df['Year'] >= 2010) & (df['Year'] <= 2021)]

# Set the style for the plot
sns.set(style="whitegrid")

# Define the years to be highlighted in red
highlight_years = [1998, 2002, 2016, 2017, 2020, 2022]

# Create a color palette with red for the highlight years and default for others
palette = {year: 'red' if year in highlight_years else 'C0' for year in filtered_df['Year'].unique()}

# Create the box plot
plt.figure(figsize=(14, 7))
sns.boxplot(x='Year', y='Var', data=filtered_df, palette=palette, order=sorted(filtered_df['Year'].unique()))

# Enhance the labels and title
plt.title('Box Plot of VAR by Year (2010-2022)', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('VAR', fontsize=14)
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability

# Show the grid for better readability
plt.grid(True, linestyle='--', alpha=0.7)

# Display the plot
plt.show()


# In[63]:


# Define the years of interest
bleaching_years = [2016, 2017, 2020, 2022]

# Filter the DataFrame to include only the specified years
bleaching_years_df = df[df['Year'].isin(bleaching_years)]


# In[64]:


#create a combined boxplot of the most recent non-bleaching years 
recent_non_bleaching_years = [2015, 2018, 2019, 2021]
filtered_df2 = df[df['Year'].isin(recent_non_bleaching_years)]
non_bleaching_years_df = filtered_df2['Var']


# In[65]:


# Create a new DataFrame for the combined data
combined_df = pd.DataFrame({
    'Year': ['Non-bleaching years'] * len(non_bleaching_years_df),  # Label these rows as 'Combined'
    'VAR': combined_data
})

# Append the combined_df to the original filtered_df
plot_df = pd.concat([bleaching_years_df, non_bleaching_years_df])

# Define your custom color palette
custom_palette = ['#4682B4', '#87CEEB', '#367588', '#B0E0E6', '#CCCCFF']  # Adjust as needed

# Create the boxplot with the custom palette, set showfliers=False to hide outliers
plt.figure(figsize=(7, 6)) 
ax = sns.boxplot(x='Year', y='Var', data=plot_df, palette=custom_palette, showfliers=False)

# Remove gridlines
ax.grid(False)

# Customize the spines to have a black border
for spine in ax.spines.values():
    spine.set_edgecolor('black')  # Set spine (border) color to black
    spine.set_linewidth(0.8)  # Adjust thickness if needed

# Add plot titles and labels
plt.title('')
plt.xlabel('')
plt.ylabel('VAR')
plt.xticks(rotation=0)  # Rotate x labels if needed

# Show the plot
plt.show()


# In[69]:


# Make Monthly Averages 
df_mo = df_ss.groupby(df_ss['Time'].dt.month)['Var'].mean().reset_index()

# Make Annual Averages over 7-year timeseries
df_yr = df_ss.groupby(pd.PeriodIndex(df_ss['Time'], freq="Y"))["Var"].mean().reset_index()

# Using pandas.Series.dt.year() & pandas.Series.dt.month() method
df_yr['Year'] = df_yr['Time'].dt.year 


# In[80]:


#Lineplot of All Monthly Means 
fig1, ax = plt.subplots(figsize = (12, 5), dpi = 250)
ax.grid(True, linewidth = 1, color = 'silver', alpha = 0.3, linestyle = '--')
plt.plot(df_mo.Time, df_mo.Var, linestyle = '-.',linewidth = 1, c='dimgrey')
plt.plot(df_mo.Time, df_mo.Var, linestyle = '', marker='s', markersize = 3, 
         c='navy',label = 'Annual Mean VAR')
# add labels
plt.xlabel('Months')
plt.ylabel('Monthly VAR')
plt.legend()
plt.show()


# In[82]:


# Lineplot of Monthly Mean throuout timeseries
plt.figure(figsize=(12, 5), dpi=250)  # Create a new figure
plt.grid(True, linewidth=1, color='silver', alpha=0.3, linestyle='--')

# Plot the data with labels for the legend
plt.plot(df_yr.Year, df_yr.Var, linestyle='-.', linewidth=1, c='dimgrey')
plt.plot(df_yr.Year, df_yr.Var, linestyle='', marker='s', markersize=3, c='navy', label='Annual Mean VAR')

# Add labels and legend
plt.xlabel('Years')
plt.ylabel('Monthly VAR')
plt.legend()  # This will now work because labels are provided

# Show the plot
plt.show()


# In[ ]:




