#!/usr/bin/env python
# coding: utf-8

# In[1]:


# imports
import pandas as pd
import datetime as dt
import numpy as np

# libraries
import seaborn as sns
from scipy.stats import gaussian_kde

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Reading the csv dataset
df = pd.read_csv (r'C:\Users\Bhavana\Desktop\econ_dataset.csv')
df.head()


# In[3]:


# Iterating the columns

for col in df.columns:
    print(col)


# In[4]:


# Get summary statistics for numeric columns
df.describe()


# In[5]:


df.STATE.unique()


# In[6]:


# There are 182 day numbers and Day 1 - 1st Oct 2008 upto Day 182 - 31st Mar 2009

start_date = dt.date(2008, 10 , 1)
number_of_days = 182

#There are 500 distinct ids hence we create the list of dates 500 times

date_list = 500*[(start_date + dt.timedelta(days = day)).isoformat() for day in range(number_of_days)]


# In[7]:


# Adding the date, month and weekday as a new columns in the dataframe

df['DATE'] = pd.to_datetime(date_list)
df['MONTH'] = df['DATE'].dt.strftime('%b')
df['WEEKDAY'] = df['DATE'].dt.day_name()
df.head(n=186)


# In[9]:


# Group by the data for plotting
df1 = df.groupby(['STATE', 'WEEKDAY'])['DAY_FREQUENCY'].sum()

# Unstack the `STATE` index, to place it as columns
df1 = df1.unstack(level='STATE')

# Define the order we want to sort the days by, create a new sorting id to sort by based on this, and then sort by that
sorter = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
sorterIndex = dict(zip(sorter,range(len(sorter))))

# Map the day_of_week to the required sort index
df1['Day_id'] = df1.index
df1['Day_id'] = df1['Day_id'].map(sorterIndex)

# Sort by the Day_id
df1.sort_values('Day_id', inplace=True)
df1 = df1.drop(['Day_id'], axis=1)
df1


# In[34]:


# plot data
fig, ax = plt.subplots(figsize=(15,7))

# setting the colors
colors=['orange', 'blue', 'purple', 'green','red']
plt.gca().set_prop_cycle(color=colors)

# edit the plot
df1.plot(ax=ax, 
         marker = 'o', 
         linestyle='dashed')

plt.title('State-Wise Frequency For Each Day', fontsize=20)
plt.xlabel('WEEKDAY', fontsize=12)
plt.ylabel('FREQUENCY', fontsize=12)
plt.grid(axis = 'x')

fig.savefig(r'C:\Users\Bhavana\Desktop\line_plot.png')   # save the figure to file


# In[11]:


df2 = df.groupby(['MONTH']).agg({'ADD_MEMBERS': "sum", 'DROPPED_MEMBERS': "sum" })

# Define the order we want to sort the days by, create a new sorting id to sort by based on this, and then sort by that
sorter = ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar']
sorterIndex = dict(zip(sorter,range(len(sorter))))

# Map the day_of_week to the required sort index
df2['Month_id'] = df2.index
df2['Month_id'] = df2['Month_id'].map(sorterIndex)

# Sort by the Day_id
df2.sort_values('Month_id', inplace=True)
df2 = df2.drop(['Month_id'], axis=1)
df2


# In[12]:


x_axis = np.arange(len(df2))


# In[33]:


# plot data
fig, ax = plt.subplots(figsize=(13,9))

plt.bar(x_axis, 
        df2.ADD_MEMBERS, 
        0.4, 
        label = 'New Members Added', 
        edgecolor='black', 
        color='green')

plt.bar(x_axis, 
        df2.DROPPED_MEMBERS, 
        0.4, 
        label = 'Members Dropped', 
        edgecolor='black', 
        color='red')

plt.title('Number Of Members Added And Dropped', fontsize=20)
plt.xticks(x_axis, df2.index)
plt.legend()
plt.show()

fig.savefig(r'C:\Users\Bhavana\Desktop\bar_plot.png')   # save the figure to file


# In[35]:


# plot data
fig, ax = plt.subplots(figsize=(15,9))


# create grouped boxplot 
sns.boxplot(x = df['NBR_OF_ADULTS'], 
            y = df['AGE'], 
            hue = df['GENDER_INPUT'], 
            palette = 'dark', 
            linewidth=1.5, 
            width=0.7)

plt.title('Number Of Adults vs. Age For Different Gender', fontsize=20)
plt.xlabel('NUMBER OF ADULTS', fontsize=12)
plt.ylabel('AGE', fontsize=12)

fig.savefig(r'C:\Users\Bhavana\Desktop\box_plot.png')   # save the figure to file


# In[ ]:




