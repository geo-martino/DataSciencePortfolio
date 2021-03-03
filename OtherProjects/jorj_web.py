#!/usr/bin/env python
# coding: utf-8

# ## Analysing web traffic data from jorjmakesmusic.com

# Obtained traffic data in .csv from wix relating to traffic from the jorjmakesmusic.com website. Cleaned data and
# showed points of interest on graphs.

# Import all necessary packages, set pandas options and open csv. All columns start with 'Traffic ' so have removed
# this from each column for better code legibility.

# In[12]:


import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

f = open('input/jorjmakesmusic_traffic.csv')
traffic_raw = pd.read_csv(f)

traffic_raw.rename(columns=lambda x: x[8:], inplace=True)

# Drop rows containing only NaN data and print how much data has been dropped/kept. Also drop columns containing
# unnecessary data that can easily be deduced from other columns.

# In[70]:


reject = traffic_raw['Page Path Ø'].isna().sum()
total = traffic_raw['Page Path Ø'].notna().sum()
traffic = traffic_raw.copy()
traffic.dropna(subset=['Page Path Ø'], inplace=True)
print('\33[91;1m', 'Insufficient data on', reject, 'visitors', '\33[0m')
print('\33[91;1m', 'Analysing', total, 'visitors', '\33[0m')

traffic.drop(['Page URL', 'Region', 'Referring URL'], axis=1, inplace=True)

# Convert remaining columns into appropriate data types.
# 
# Session duration and page time columns are in units of days, convert to seconds and update column names.
# 
# Set 'Day' as index and sort by date.
# 
# Assign 'max_date' as the date at which the data was obtained.

# In[71]:


num = ['Page Views', 'Site Sessions Ø', 'Unique Visitors', 'Site Bounce Rate Ø',
       'Avg. Session Duration Ø', 'Avg. Time on Page Ø']
bina = ['Is the Last Page Viewed Ø (Yes / No)', 'Is the First Page Viewed Ø (Yes / No)']
time = ['Avg. Session Duration Ø', 'Avg. Time on Page Ø']

traffic[num] = traffic[num].apply(pd.to_numeric)
traffic['Avg. Pages per Session Ø'] = traffic['Avg. Pages per Session Ø'].astype('int64')
traffic[time] = traffic[time].apply(lambda x: x * 24 * 60 * 60)
traffic.rename(columns=lambda x: x[:-2] + ' (s)' if x in time else x, inplace=True)

for i in bina:
    d = {'No': 0, 'Yes': 1}
    traffic[i] = traffic[i].map(d)
    traffic[i] = traffic[i].astype('bool')
    traffic.rename(columns={i: i[7:-13]}, inplace=True)

traffic['Day'] = pd.to_datetime(traffic['Day'], format='%d/%m/%Y')
traffic = traffic.set_index('Day').sort_index()

max_date = str(max(traffic.index))[0:10]

# Plot data with seaborn.

# In[85]:


sns.set()
plt.figure(figsize=(6, 6))
ax = sns.countplot(x="Device Type", data=traffic)
ax.set(xlabel='Device Type', ylabel='Visitors', title='Visitors per device type for jorjmakesmusic.com')

plt.show()

# In[84]:


plt.figure(figsize=(16, 6))
ax = sns.countplot(x="Country", data=traffic)
ax.set(xlabel='Country', ylabel='Visitors', title='Visitors per Country for jorjmakesmusic.com')

plt.xticks(rotation=45)
plt.show()

# In[83]:


ax = sns.countplot(x="New or Returning Visitor", data=traffic)
ax.set(xlabel=None, ylabel='Visitors', title='New or Returning Visitor?')

plt.show()

# Website released 05/05/2020, any data prior to this is therefore of no relevance therefore, data rejected for
# 'visitor rate per day' graphs.

# In[78]:


dayData = traffic.index.value_counts().sort_index()['2020-05-05':]

plt.figure(figsize=(16, 6))
ax = sns.lineplot(data=dayData)
ax.set(xlabel='Date', ylabel='Visitors', title='Visitors per Day for jorjmakesmusic.com')
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))

plt.show()

# In[77]:


dates = traffic.index.floor('D')
sessionTime = traffic.groupby(dates)['Avg. Session Duration (s)'].sum()

plt.figure(figsize=(16, 6))
ax = sns.lineplot(data=sessionTime)
ax.set(xlabel='Date', ylabel='Time (s)', title='Average Session Duration per day for jorjmakesmusic.com')
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))

plt.show()

# In[76]:


dates = traffic.index.floor('D')
pageTime = traffic.groupby(dates)['Avg. Time on Page (s)'].sum()

plt.figure(figsize=(16, 6))
ax = sns.lineplot(data=pageTime)
ax.set(xlabel='Date', ylabel='Time (s)', title='Average Time on Page per day for jorjmakesmusic.com')
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))

plt.show()
