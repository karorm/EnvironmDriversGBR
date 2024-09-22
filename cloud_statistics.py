#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from scipy.stats import ttest_ind


# In[2]:


# Load the data
df = pd.read_csv('cloud_data.csv')


# In[3]:


print(df.head())


# In[4]:


print(df)


# In[5]:


# Check data types and general info
print(df.info())


# In[6]:


import pandas as pd
from scipy.stats import shapiro, mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# Load the data
df = pd.read_csv('cloud_data.csv')

# Check column names for leading/trailing spaces and strip them if necessary
df.columns = df.columns.str.strip()

# Separate the data by temperature class
max_temp_data = df[df['Temperature_class'].str.strip() == 'max']['Cloud_cover']
min_temp_data = df[df['Temperature_class'].str.strip() == 'min']['Cloud_cover']

# Check the size of each group
print(f'Number of data points for Max Temperature Cloud Cover: {len(max_temp_data)}')
print(f'Number of data points for Min Temperature Cloud Cover: {len(min_temp_data)}')

# Shapiro-Wilk Test for normality if sufficient data
if len(max_temp_data) >= 3 and len(min_temp_data) >= 3:
    max_stat, max_p_value = shapiro(max_temp_data)
    min_stat, min_p_value = shapiro(min_temp_data)
    
    print(f'Shapiro-Wilk Test for Max Temperature Cloud Cover: Statistic={max_stat}, p-value={max_p_value}')
    print(f'Shapiro-Wilk Test for Min Temperature Cloud Cover: Statistic={min_stat}, p-value={min_p_value}')
else:
    print("Not enough data to perform Shapiro-Wilk test.")

# Histogram and Q-Q plot for Max Temperature Cloud Cover
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.histplot(max_temp_data, kde=True)
plt.title('Histogram of Max Temperature Cloud Cover')

plt.subplot(1, 2, 2)
stats.probplot(max_temp_data, dist="norm", plot=plt)
plt.title('Q-Q Plot of Max Temperature Cloud Cover')

plt.tight_layout()
plt.show()

# Histogram and Q-Q plot for Min Temperature Cloud Cover
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.histplot(min_temp_data, kde=True)
plt.title('Histogram of Min Temperature Cloud Cover')

plt.subplot(1, 2, 2)
stats.probplot(min_temp_data, dist="norm", plot=plt)
plt.title('Q-Q Plot of Min Temperature Cloud Cover')

plt.tight_layout()
plt.show()

# Perform Mann-Whitney U Test if needed
u_stat, u_p_value = mannwhitneyu(max_temp_data, min_temp_data, alternative='two-sided')
print(f'Mann-Whitney U Test: U-statistic = {u_stat}, p-value = {u_p_value}')


# In[7]:


from scipy.stats import shapiro, levene

# Shapiro-Wilk Test for normality if sufficient data
if len(max_temp_data) >= 3 and len(min_temp_data) >= 3:
    max_stat, max_p_value = shapiro(max_temp_data)
    min_stat, min_p_value = shapiro(min_temp_data)
    
    print(f'Shapiro-Wilk Test for Max Temperature Cloud Cover: Statistic={max_stat:.4f}, p-value={max_p_value:.4f}')
    print(f'Shapiro-Wilk Test for Min Temperature Cloud Cover: Statistic={min_stat:.4f}, p-value={min_p_value:.4f}')
    
    if max_p_value < 0.05:
        print("Max Temperature Cloud Cover data is not normally distributed.")
    else:
        print("Max Temperature Cloud Cover data is normally distributed.")
        
    if min_p_value < 0.05:
        print("Min Temperature Cloud Cover data is not normally distributed.")
    else:
        print("Min Temperature Cloud Cover data is normally distributed.")

    # Levene's Test for equality of variances
    levene_stat, levene_p_value = levene(max_temp_data, min_temp_data)
    print(f'Levene\'s Test: Statistic={levene_stat:.4f}, p-value={levene_p_value:.4f}')
    
    if levene_p_value < 0.05:
        print("Variances are significantly different.")
    else:
        print("Variances are not significantly different.")
else:
    print("Not enough data to perform Shapiro-Wilk test.")


# In[8]:


from scipy.stats import ttest_ind

# Perform t-test
t_stat, p_val = ttest_ind(max_temp_data, min_temp_data, equal_var=False)
print(f'T-test: t-statistic = {t_stat}, p-value = {p_val}')


# In[9]:


# Separate the data into specific years and other years
specific_years = df[df['aus_summer'].isin(['15/16', '16/17', '19/20', '21/22'])]
other_years = df[~df['aus_summer'].isin(['15/16', '16/17', '19/20', '21/22'])]

specific_cloud_cover = specific_years['Cloud_cover']
other_cloud_cover = other_years['Cloud_cover']

# Perform t-test
t_stat, p_val = ttest_ind(specific_cloud_cover, other_cloud_cover, equal_var=False)
print(f'T-test: t-statistic = {t_stat}, p-value = {p_val}')


# In[10]:


import pandas as pd
from scipy.stats import ttest_ind

# Sample ENSO classification for years based on the provided list
enso_classification = {
    '12/13': 'Neutral',
    '13/14': 'Neutral',
    '14/15': 'Neutral',
    '15/16': 'El Niño',
    '16/17': 'Neutral',
    '17/18': 'Neutral',
    '18/19': 'Neutral',
    '19/20': 'Neutral',
    '20/21': 'Neutral',
    '21/22': 'La Niña'
}

# Add ENSO classification to your dataframe
df['ENSO'] = df['aus_summer'].map(enso_classification)

# Separate the data into El Niño and La Niña years
el_nino_years = df[df['ENSO'] == 'El Niño']
la_nina_years = df[df['ENSO'] == 'La Niña']

# Extract the cloud cover data for both groups
el_nino_cloud_cover = el_nino_years['Cloud_cover']
la_nina_cloud_cover = la_nina_years['Cloud_cover']

# Perform t-test
t_stat, p_val = ttest_ind(el_nino_cloud_cover, la_nina_cloud_cover, equal_var=False)
print(f'T-test: t-statistic = {t_stat}, p-value = {p_val}')

