#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd

# Read the data from the txt file into a DataFrame
df = pd.read_csv('data.txt', delim_whitespace=True)

# Convert 'time' to datetime format (if not already in datetime)
df['time'] = pd.to_datetime(df['time'])

# Display the first few rows of the DataFrame to verify
print(df)


# In[3]:


import pandas as pd
from scipy.stats import shapiro, mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# Assuming df is your DataFrame containing the data
# Separate the data by temperature type
sst_data = df['SST_mo_comp']
ssh_data = df['SSH_mo_comp']
chl_a_data = df['Chl_a']

# Check the size of each group
print(f'Number of data points for SST: {len(sst_data)}')
print(f'Number of data points for SSH: {len(ssh_data)}')
print(f'Number of data points for Chl_a: {len(chl_a_data)}')

# Shapiro-Wilk Test for normality if sufficient data
for data, name in zip([sst_data, ssh_data, chl_a_data], ['SST', 'SSH', 'Chl_a']):
    if len(data) >= 3:
        stat, p_value = shapiro(data)
        print(f'Shapiro-Wilk Test for {name}: Statistic={stat}, p-value={p_value}')
    else:
        print(f"Not enough data to perform Shapiro-Wilk test for {name}.")

# Function to plot histogram and Q-Q plot
def plot_data(data, title):
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    sns.histplot(data, kde=True)
    plt.title(f'Histogram of {title}')

    plt.subplot(1, 2, 2)
    stats.probplot(data, dist="norm", plot=plt)
    plt.title(f'Q-Q Plot of {title}')

    plt.tight_layout()
    plt.show()

# Plot for SST, SSH, and Chl_a
plot_data(sst_data, 'SST')
plot_data(ssh_data, 'SSH')
plot_data(chl_a_data, 'Chl_a')

# Perform Mann-Whitney U Test between pairs
u_stat_sst_ssh, u_p_value_sst_ssh = mannwhitneyu(sst_data, ssh_data, alternative='two-sided')
print(f'Mann-Whitney U Test between SST and SSH: U-statistic = {u_stat_sst_ssh}, p-value = {u_p_value_sst_ssh}')

u_stat_sst_chl, u_p_value_sst_chl = mannwhitneyu(sst_data, chl_a_data, alternative='two-sided')
print(f'Mann-Whitney U Test between SST and Chl_a: U-statistic = {u_stat_sst_chl}, p-value = {u_p_value_sst_chl}')

u_stat_ssh_chl, u_p_value_ssh_chl = mannwhitneyu(ssh_data, chl_a_data, alternative='two-sided')
print(f'Mann-Whitney U Test between SSH and Chl_a: U-statistic = {u_stat_ssh_chl}, p-value = {u_p_value_ssh_chl}')


# In[4]:


from scipy.stats import kstest, norm

# Perform Kolmogorov-Smirnov test for 'SST' variable
statistic_sst, p_value_sst = kstest(df['SST_mo_comp'], 'norm')
print(f"Kolmogorov-Smirnov test statistic for SST: {statistic_sst}, p-value: {p_value_sst}")

if p_value_sst > 0.05:
    print("SST data is normally distributed (fail to reject H0)")
else:
    print("SST data is not normally distributed (reject H0)")

# Perform Kolmogorov-Smirnov test for 'SSH' variable
statistic_ssh, p_value_ssh = kstest(df['SSH_mo_comp'], 'norm')
print(f"Kolmogorov-Smirnov test statistic for SSH: {statistic_ssh}, p-value: {p_value_ssh}")

if p_value_ssh > 0.05:
    print("SSH data is normally distributed (fail to reject H0)")
else:
    print("SSH data is not normally distributed (reject H0)")

# Perform Kolmogorov-Smirnov test for 'SSH' variable
statistic_ssh, p_value_ssh = kstest(df['Chl_a'], 'norm')
print(f"Kolmogorov-Smirnov test statistic for SSH: {statistic_ssh}, p-value: {p_value_ssh}")

if p_value_ssh > 0.05:
    print("chl data is normally distributed (fail to reject H0)")
else:
    print("chl data is not normally distributed (reject H0)")


# In[5]:


import pandas as pd
from scipy.stats import shapiro, levene

# Assuming df is your DataFrame with the relevant columns

# Perform Shapiro-Wilk test for normality
statistic_sst, p_value_sst = shapiro(df['SST_mo_comp'])
statistic_ssh, p_value_ssh = shapiro(df['SSH_mo_comp'])
statistic_chl, p_value_chl = shapiro(df['Chl_a'])

# Print results for Shapiro-Wilk tests
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

print(f"\nShapiro-Wilk test for Chl_a:")
print(f"  Statistic: {statistic_chl}, p-value: {p_value_chl}")
if p_value_chl > 0.05:
    print("  Conclusion: Chl_a data is normally distributed (fail to reject H0)")
else:
    print("  Conclusion: Chl_a data is not normally distributed (reject H0)")

# Perform Levene's test for equal variances
levene_stat, levene_p_value = levene(df['SST_mo_comp'], df['SSH_mo_comp'], df['Chl_a'])

print(f"\nLevene's test for equal variances:")
print(f"  Statistic: {levene_stat}, p-value: {levene_p_value}")
if levene_p_value > 0.05:
    print("  Conclusion: Variances are equal (fail to reject H0)")
else:
    print("  Conclusion: Variances are not equal (reject H0)")


# In[6]:


# Exclude the 'time' column for correlation calculation
df_numeric = df.drop(columns=['time'])

# Calculate the Spearman correlation matrix
spearman_corr = df_numeric.corr(method='spearman')

# Display the Spearman correlation matrix
print(spearman_corr)

# Set up the matplotlib figure
plt.figure(figsize=(6, 4))

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(spearman_corr, annot=True, cmap='viridis', vmin=-1, vmax=1, center=0)

# Add titles and labels
plt.title('Spearman Rank Correlation Matrix')
plt.show()


# In[7]:


import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Fit the ANOVA model (using Chl_a as the dependent variable)
model = ols('Chl_a ~ SST_mo_comp + SSH_mo_comp', data=df).fit()

# Perform ANOVA
anova_table = sm.stats.anova_lm(model, typ=2)

# Display the ANOVA table
print(anova_table)


# In[8]:


# Extract R-squared value
r_squared = model.rsquared
print(f'R-squared: {r_squared:.4f}')

