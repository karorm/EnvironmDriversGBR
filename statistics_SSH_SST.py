#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# Read the data from the txt file into a DataFrame
df = pd.read_csv('data.txt', delim_whitespace=True)

# Convert 'time' to datetime format (if not already in datetime)
df['time'] = pd.to_datetime(df['time'])

# Display the first few rows of the DataFrame to verify
print(df.head())


# In[2]:


import pandas as pd
from scipy.stats import shapiro, mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# Separate the data by temperature type
sst_data = df['SST']
ssh_data = df['SSH']

# Check the size of each group
print(f'Number of data points for SST: {len(sst_data)}')
print(f'Number of data points for SSH: {len(ssh_data)}')

# Shapiro-Wilk Test for normality if sufficient data
if len(sst_data) >= 3 and len(ssh_data) >= 3:
    sst_stat, sst_p_value = shapiro(sst_data)
    ssh_stat, ssh_p_value = shapiro(ssh_data)
    
    print(f'Shapiro-Wilk Test for SST: Statistic={sst_stat}, p-value={sst_p_value}')
    print(f'Shapiro-Wilk Test for SSH: Statistic={ssh_stat}, p-value={ssh_p_value}')
else:
    print("Not enough data to perform Shapiro-Wilk test.")

# Histogram and Q-Q plot for SST
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.histplot(sst_data, kde=True)
plt.title('Histogram of SST')

plt.subplot(1, 2, 2)
stats.probplot(sst_data, dist="norm", plot=plt)
plt.title('Q-Q Plot of SST')

plt.tight_layout()
plt.show()

# Histogram and Q-Q plot for SSH
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.histplot(ssh_data, kde=True)
plt.title('Histogram of SSH')

plt.subplot(1, 2, 2)
stats.probplot(ssh_data, dist="norm", plot=plt)
plt.title('Q-Q Plot of SSH')

plt.tight_layout()
plt.show()

# Perform Mann-Whitney U Test if needed
u_stat, u_p_value = mannwhitneyu(sst_data, ssh_data, alternative='two-sided')
print(f'Mann-Whitney U Test: U-statistic = {u_stat}, p-value = {u_p_value}')


# In[3]:


from scipy.stats import kstest, norm

# Perform Kolmogorov-Smirnov test for 'SST' variable
statistic_sst, p_value_sst = kstest(df['SST'], 'norm')
print(f"Kolmogorov-Smirnov test statistic for SST: {statistic_sst}, p-value: {p_value_sst}")

if p_value_sst > 0.05:
    print("SST data is normally distributed (fail to reject H0)")
else:
    print("SST data is not normally distributed (reject H0)")

# Perform Kolmogorov-Smirnov test for 'SSH' variable
statistic_ssh, p_value_ssh = kstest(df['SSH'], 'norm')
print(f"Kolmogorov-Smirnov test statistic for SSH: {statistic_ssh}, p-value: {p_value_ssh}")

if p_value_ssh > 0.05:
    print("SSH data is normally distributed (fail to reject H0)")
else:
    print("SSH data is not normally distributed (reject H0)")


# In[4]:


from scipy.stats import shapiro, levene

# Perform Shapiro-Wilk test for normality
statistic_sst, p_value_sst = shapiro(df['SST'])
statistic_ssh, p_value_ssh = shapiro(df['SSH'])

print(f"Shapiro-Wilk test for SST:")
print(f"  Statistic: {statistic_sst}, p-value: {p_value_sst}")
if p_value_sst > 0.05:
    print("  Conclusion: SST data is normally distributed (fail to reject H0)")
else:
    print("  Conclusion: SST data is not normally distributed (reject H0)")

print(f"\nShapiro-Wilk test for SSH:")
print(f"  Statistic: {statistic_ssh}, p-value: {p_value_ssh}")
if p_value_ssh > 0.05:
    print("  Conclusion: SSH data is normally distributed (fail to reject H0)")
else:
    print("  Conclusion: SSH data is not normally distributed (reject H0)")

# Perform Levene's Test for equality of variances
levene_statistic, levene_p_value = levene(df['SST'], df['SSH'])
print(f"\nLevene's Test for equality of variances:")
print(f"  Statistic: {levene_statistic}, p-value: {levene_p_value}")
if levene_p_value > 0.05:
    print("  Conclusion: Variances are not significantly different (fail to reject H0)")
else:
    print("  Conclusion: Variances are significantly different (reject H0)")


# In[5]:


from scipy.stats import shapiro

# Perform Shapiro-Wilk test for normality
statistic_sst, p_value_sst = shapiro(df['SST'])
statistic_ssh, p_value_ssh = shapiro(df['SSH'])

print(f"Shapiro-Wilk test for SST:")
print(f"  Statistic: {statistic_sst}, p-value: {p_value_sst}")
if p_value_sst > 0.05:
    print("  Conclusion: SST data is normally distributed (fail to reject H0)")
else:
    print("  Conclusion: SST data is not normally distributed (reject H0)")

print(f"\nShapiro-Wilk test for SSH:")
print(f"  Statistic: {statistic_ssh}, p-value: {p_value_ssh}")
if p_value_ssh > 0.05:
    print("  Conclusion: SSH data is normally distributed (fail to reject H0)")
else:
    print("  Conclusion: SSH data is not normally distributed (reject H0)")


# In[6]:


from scipy.stats import spearmanr

# Perform Spearman rank correlation
spearman_corr, p_value = spearmanr(df['SST'], df['SSH'])

print(f"Spearman Correlation: {spearman_corr}, P-value: {p_value}")


# In[7]:


correlation_kendall = df['SST'].corr(df['SSH'], method='kendall')
print(f"Kendall Tau correlation coefficient: {correlation_kendall}")


# In[8]:


#technically not relevant becasue my data is not normally distributed 
# Calculate Pearson correlation coefficient
correlation_pearson = df['SST'].corr(df['SSH'])
print(f"Pearson correlation coefficient: {correlation_pearson}")


# In[9]:


import numpy as np

# Assuming df is your DataFrame with 'SST' and 'SSH' columns

# Apply natural logarithm transformation
df['SST_log'] = np.log(df['SST'])
df['SSH_log'] = np.log(df['SSH'])

# Display the transformed DataFrame
print(df.head())


# In[10]:


# Shapiro-Wilk test for normality
statistic_sst, p_value_sst = shapiro(df['SST_log'])
statistic_ssh, p_value_ssh = shapiro(df['SSH_log'])

print(f"Shapiro-Wilk test for SST_log:")
print(f"  Statistic: {statistic_sst}, p-value: {p_value_sst}")
if p_value_sst > 0.05:
    print("  Conclusion: SST_log data is normally distributed (fail to reject H0)")
else:
    print("  Conclusion: SST_log data is not normally distributed (reject H0)")

print(f"\nShapiro-Wilk test for SSH_log:")
print(f"  Statistic: {statistic_ssh}, p-value: {p_value_ssh}")
if p_value_ssh > 0.05:
    print("  Conclusion: SSH_log data is normally distributed (fail to reject H0)")
else:
    print("  Conclusion: SSH_log data is not normally distributed (reject H0)")


# In[11]:


# Apply base 10 logarithm transformation
df['SST_log10'] = np.log10(df['SST'])
df['SSH_log10'] = np.log10(df['SSH'])

# Display the transformed DataFrame
print(df.head())


# In[12]:


# Shapiro-Wilk test for normality
statistic_sst, p_value_sst = shapiro(df['SST_log10'])
statistic_ssh, p_value_ssh = shapiro(df['SSH_log10'])

print(f"Shapiro-Wilk test for SST_log:")
print(f"  Statistic: {statistic_sst}, p-value: {p_value_sst}")
if p_value_sst > 0.05:
    print("  Conclusion: SST_log10 data is normally distributed (fail to reject H0)")
else:
    print("  Conclusion: SST_log10 data is not normally distributed (reject H0)")

print(f"\nShapiro-Wilk test for SSH_log:")
print(f"  Statistic: {statistic_ssh}, p-value: {p_value_ssh}")
if p_value_ssh > 0.05:
    print("  Conclusion: SSH_log10 data is normally distributed (fail to reject H0)")
else:
    print("  Conclusion: SSH_log10 data is not normally distributed (reject H0)")


# In[14]:


import statsmodels.api as sm
from statsmodels.formula.api import ols
# Fit the linear regression model
model = ols('SSH ~ SST', data=df).fit()

# Perform ANOVA on the fitted linear model
anova_table = sm.stats.anova_lm(model, typ=2)

# Display the ANOVA table
print(anova_table)


# In[15]:


# Get R-squared value
r_squared = model.rsquared

# Display the R-squared value
print(f"R-squared: {r_squared:.4f}")


# In[ ]:




