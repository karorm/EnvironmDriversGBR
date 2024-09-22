#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pyhdf.SD import SD, SDC

import matplotlib.pyplot as plt
from matplotlib import gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.geometry import LinearRing


import numpy as np


# In[2]:


file_name = 'cloud_data.hdf'
hdf = SD(file_name, SDC.READ)

# List all datasets in the file
datasets_dict = hdf.datasets()

# Print the names of the datasets and their dimensions
for dataset in datasets_dict.keys():
    print(f"{dataset}")


# In[3]:


# Extract your variables:
var = hdf.select('Cloud_Fraction_Mean_Mean')[:] # Month
#var = hdf.select('Cloud_Fraction_Mean')[:]  # Daily Data
lat = hdf.select('YDim')[:]
lon = hdf.select('XDim')[:]


# In[4]:


# Check dimensions and shapes
print(f"Data_Var shape: {var.shape}")
print(f"Latitude shape: {lat.shape}")
print(f"Longitude shape:{lon.shape}")


# In[5]:


# Create a meshgrid for latitude and longitude
lon, lat = np.meshgrid(lon, lat)


# In[6]:


plt.imshow(var)


# In[7]:


# Subset using Rows and Column Values from above Plot
row1 = 110
row2 = 117
col1 = 330
col2 = 340


# In[8]:


# Now Subset your Data to speed up Processing and Plotting
var_ss = var[row1:row2, col1:col2]
lat_ss = lat[row1:row2, col1:col2]
lon_ss = lon[row1:row2, col1:col2]


# In[9]:


# Define your min and max values
# divided by 100 for cloud % 
min_cf=np.nanmin(var_ss)/100
max_cf=np.nanmax(var_ss)/100
mean_cf=np.mean(var_ss)/100

print(min_cf)
print(max_cf)
print(mean_cf)


# In[12]:


# Define the grid indices corresponding to the area of interest
grid_index_lat = 3
grid_index_lon = 1

# Plot your data
fig1 = plt.figure(figsize=(17, 17), dpi=300)
gs = gridspec.GridSpec(1, 1)

# Set data projection and request output projection
data_projection = ccrs.PlateCarree()
output_projection = ccrs.PlateCarree()
land_resolution = '10m'
land_poly = cfeature.NaturalEarthFeature('physical', 'land', "10m",
                                         edgecolor='k',
                                         facecolor='silver')
ax = plt.subplot(gs[0, 0], projection=ccrs.PlateCarree(central_longitude=0.0))

# Plot your variable with pre-defined min and max values, bone_reverse colorscale
im = plt.pcolormesh(lon_ss, lat_ss, (var_ss / 100), vmin=min_cf, vmax=max_cf,
                    cmap=plt.cm.bone_r)

# Mapping
ax.coastlines(resolution = land_resolution, color = 'black', linewidth = 1)
ax.add_feature(land_poly)
g1 = ax.gridlines(draw_labels = True, linewidth = 1, color = 'lightgray', alpha = 0.5, linestyle = '--')
g1.top_labels  = False
g1.right_labels= False
g1.xlabel_style = {'size': 11, 'color': 'gray'}
g1.ylabel_style = {'size': 11, 'color': 'gray'}
# Colourbar
cbar = plt.colorbar(orientation = "horizontal", fraction = 0.1, pad = 0.04) 
cbar.set_label('8-day Mean Cloud Fraction', fontsize = 13)

# adding subset box
lons = [152.36,151.36,151.36,152.36]
lats = [-23.08, -23.08,-23.6,-23.6]
ring = LinearRing(list(zip(lons, lats)))
# Plot the LinearRing with a specific color (e.g., blue)
x, y = ring.xy
plt.plot(x, y, color='red')  # Change 'blue' to the desired color name or code

plt.show()


# In[11]:


# Define the grid indices corresponding to the pixel of interest where CBG reefs are 
grid_index_lat = 3
grid_index_lon = 1

# Extract the pixel values at the specified grid indices and the pixel to the right
pixel_value_left = var_ss[grid_index_lat, grid_index_lon] / 100
pixel_value_right = var_ss[grid_index_lat, grid_index_lon + 1] / 100

# Calculate the mean of the two pixel values
mean_value = (pixel_value_left + pixel_value_right) / 2

print("Mean value of pixel at grid indices ({}, {}) and its right neighbor:".format(grid_index_lat, grid_index_lon), mean_value)


# In[ ]:





# In[ ]:




