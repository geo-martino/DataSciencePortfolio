import json

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def jprint(obj):
    # create a formatted string of the Python JSON object
    text = json.dumps(obj, sort_keys=True, indent=4)
    print(text)


def open_df(filename):
    # open json file and define as 'data' and arrange into dataframe
    with open("input/" + filename + ".json") as f:
        data = [json.loads(line) for line in f]
    return pd.DataFrame(data)


def average_cases(raw, days):
    """
    :param raw: pandas dataframe with covid data to average
    :param days: amount of days to average cases over
    :return: pandas dataframe containing date, average rates, countries and population
    """
    raw = raw.reset_index()
    country = ''
    i = 0
    avg = pd.DataFrame(columns=['date', 'rate', 'countries', 'pop'])

    while i < len(raw):
        if raw['countries'][i] == country:
            average = sum(raw['rate'][i - days:i]) / days
            # print(raw['date'][i], raw['countries'][i], average)
            avg = avg.append({'date': raw['date'][i], 'rate': average,
                              'countries': raw['countries'][i], 'pop': raw['pop'][i]},
                             ignore_index=True)
            i += 1
        else:
            i += days
            country = raw['countries'][i]

    return avg


# drop NaN data, rename columns, convert data types, remove _ from country names
df = open_df("covid_daily")
df.dropna(axis=0, inplace=True)
df.rename({'population': 'pop', 'daily_confirmed_cases': 'daily', 'confirmed_cases': 'total',
           'countries_and_territories': 'countries', 'geo_id': 'code'}, axis=1, inplace=True)
cols = ['pop', 'daily', 'total']
df[cols] = df[cols].apply(pd.to_numeric)
df['date'] = pd.to_datetime(df['date'])
df['countries'] = df['countries'].str.replace('_', ' ')

# show user country list with codes and get user input for country codes
print(df.drop_duplicates(subset=['code'])[['countries', 'code']].to_string(index=False))
codes = []
input_str = input('Enter country code: ')
while input_str != '':
    codes.append(input_str)
    input_str = input('Enter country code (hit return with no entry to finish): ')

# reduce data based on user input for country codes
df = df[df.isin(codes).any(1)]

# create data for infections per #
per = 100000
df['rate'] = (df['daily'] / df['pop']) * (per * 10)

# get user input for rolling average and define per # of population
days = 0
while days <= 0:
    i = input("Enter amount of days for rolling average: ")
    try:
        days = int(i)
    except ValueError:
        print('\33[91;1m', "Error: Enter an integer greater than 0", '\33[0m')
        days = 0

# average daily cases based on user input
avg_data = average_cases(df, days)
max_date = str(max(df['date']))[0:10]

# plot
sns.set()
fig, ax = plt.subplots(2, sharex=True, figsize=(12, 10), num='Covid-19 Data (accurate as of ' + max_date + ')',
                       facecolor='lightgrey')
ax[0] = sns.lineplot(data=avg_data, hue='countries', x='date', y='rate', ax=ax[0])
ax[0].legend(title=False, fancybox=True, framealpha=1, shadow=True, borderpad=1)
ax[0].set(xlabel='Month', ylabel='Confirmed cases per ' + str(per), title='Confirmed cases per 100000 over '
                                                                          + str(days) + ' day rolling average')
ax[0].xaxis.set_major_formatter(mdates.DateFormatter("%b"))

ax[1] = sns.lineplot(data=df, hue='countries', x='date', y='total', ax=ax[1])
ax[1].legend(title=False, fancybox=True, framealpha=1, shadow=True, borderpad=1)
ax[1].set(xlabel='Month', ylabel='Total confirmed cases', title='Total confirmed cases by country')
ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%b"))

plt.xticks(rotation=45)
plt.savefig('output/covid_' + max_date + '.png')
plt.show()
