#!/usr/bin/env python
# coding: utf-8

# In[27]:


get_ipython().run_line_magic('matplotlib', 'inline')
# Mapping
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.dates as dates
from matplotlib.dates import DateFormatter
# Data
import netCDF4 as nc
import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')


# In[28]:


pwd # Confirming your current path


# In[40]:


# Replace my path with pathname
DATA_PATH = os.path.join("/path/to/data")
INFILE = "data.nc" # replace with filename


# In[30]:


nc_ = nc.Dataset(os.path.join(DATA_PATH, INFILE))
nc_


# In[31]:


nc_.variables['zos'].dimensions


# In[32]:


# Pull out your SST / SSH and co-variables 
# (this step may take time if your dataset is big)
var = nc_.variables['zos'][:]                     # [:,0,:,:] if you need to remove depth dimension, else [:]
time= nc_.variables['time']
lat = nc_.variables['latitude'][:]
lon = nc_.variables['longitude'][:]
print('Number of timesteps, lat, and lon =',var.shape)


# In[33]:


# Get your dates into a user-friendly format: dtime is your new date variable
caldr = time.getncattr('calendar')
units = time.getncattr('units')
dtime = nc.num2date(time[:], units, calendar = caldr, only_use_cftime_datetimes = False)


# In[34]:


LAT_ss = [-23.6, -23.08]
LON_ss = [151.36, 152.36]
lat_min_index = np.argmin(np.abs(lat - LAT_ss[0]))
lat_max_index = np.argmin(np.abs(lat - LAT_ss[1]))
lon_min_index = np.argmin(np.abs(lon - LON_ss[0]))
lon_max_index = np.argmin(np.abs(lon - LON_ss[1]))


# In[35]:


var_ss = []
timesteps = var.shape[0]
for x in range(timesteps):
    subset = np.nanmean(var[x,lat_min_index:lat_max_index,lon_min_index:lon_max_index])
    var_ss.append(subset)


# In[36]:


df_ss = pd.DataFrame({'Var':var_ss, 'Time':dtime})
print(df_ss)


# In[37]:


# Find first September within df to visualise potential Eddy activity during austral spring
# Can be adapted to find min and max values per year/month etc
# Filter and get the first 'Var' for September of each year without creating a new DataFrame
first_sept_per_year = df_ss[df_ss['Time'].dt.month.eq(9)] \
    .loc[lambda x: x.groupby(x['Time'].dt.year)['Time'].idxmin()]

# Display the result
print(first_sept_per_year)


# In[38]:


import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.gridspec import GridSpec

# Define your starting index and the number of plots
starting_index = 10105
number_of_plots = 12

# Generate indices starting from the starting_index
indices = [starting_index + i for i in range(number_of_plots)]

# Create a figure with GridSpec layout
fig = plt.figure(figsize=(20, 20))
gs = GridSpec(3, 4, figure=fig, wspace=0.3, hspace=0.1)  # Reduce hspace to make rows closer together

land_resolution = '10m'
data_projection = ccrs.PlateCarree()
output_projection = ccrs.PlateCarree()
land_poly = cfeature.NaturalEarthFeature('physical', 'land', "10m",
                                         edgecolor='k',
                                         facecolor='black')

# Loop over the indices to create subplots
for i, index in enumerate(indices):
    row, col = divmod(i, 4)
    ax = plt.subplot(gs[row, col], projection=ccrs.PlateCarree(central_longitude=0.0))
    im = ax.pcolormesh(lon, lat, var[index], cmap=plt.cm.viridis)
    im.set_clim(0.4, 0.7) #adjust for better visualisation
    ax.coastlines(resolution=land_resolution, color='black', linewidth=1)
    g1 = ax.gridlines(draw_labels=True, linewidth=1, color='lightgray', alpha=0.2, linestyle='--')
    g1.top_labels = False
    g1.right_labels = False
    date_label = df_ss.loc[index, 'Time'].strftime('%Y-%m-%d')
    cbar = plt.colorbar(im, ax=ax, orientation="horizontal", fraction=0.046, pad=0.04)
    cbar.set_label(date_label, fontsize=12, weight='bold')  # Label the plot with its date

plt.tight_layout()
plt.show()


# In[39]:


import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.gridspec import GridSpec

# Define your starting index and the number of plots
starting_index = 10105
number_of_plots = 4

# Generate indices starting from the starting_index
indices = [starting_index + i for i in range(number_of_plots)]

# Create a figure with GridSpec layout for 2x2 arrangement
fig = plt.figure(figsize=(20, 20))
gs = GridSpec(2, 2, figure=fig, wspace=0.05, hspace=0.05)  # Minimal space between subplots

land_resolution = '10m'
data_projection = ccrs.PlateCarree()
output_projection = ccrs.PlateCarree()
land_poly = cfeature.NaturalEarthFeature('physical', 'land', "10m",
                                         edgecolor='k',
                                         facecolor='black')

# Loop over the indices to create subplots in a 2x2 grid
for i, index in enumerate(indices):
    row, col = divmod(i, 2)  # Adjust to handle 2x2 grid
    ax = plt.subplot(gs[row, col], projection=ccrs.PlateCarree(central_longitude=0.0))
    
    # Plot data on each axis
    im = ax.pcolormesh(lon, lat, var[index], cmap=plt.cm.viridis)
    im.set_clim(0.35, 0.65) 
    ax.coastlines(resolution=land_resolution, color='black', linewidth=1)
    
    # Adjust the gridlines
    g1 = ax.gridlines(draw_labels=True, linewidth=1, color='lightgray', alpha=0.2, linestyle='--')
    g1.top_labels = False
    g1.right_labels = False
    
    # Get the date for the plot
    date_label = df_ss.loc[index, 'Time'].strftime('%Y-%m-%d')
    
    # Manually adjust subplot position (to bring rows closer)
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0 + 0.05, pos.width, pos.height])  # Adjust the y-position of the subplot
    
    # Manually add colorbars outside of the plot
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.046, pad=0.02)  # Reduced pad
    cbar.set_label(date_label, fontsize=12, weight='bold')

plt.show()


# In[ ]:




