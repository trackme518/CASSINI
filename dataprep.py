import pandas as pd
import numpy as np
import os
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score

import os
file_path = os.path.dirname(os.path.realpath(__file__))
file_path = os.path.join(file_path, '')
print(file_path)

#--------------------------------------------
## Read weather data
# https://www.copernicus.eu/en/copernicus-services
pd.set_option('display.max.columns', 100)

## set the period of forcast in 2021
start_mth = datetime.datetime(2021, 1, 1)
end_mth = datetime.datetime(2021, 12, 31)

## Read Weather Data
weather = pd.read_csv(file_path+'weather_data.csv', encoding='utf-8')
print(weather.head())

# Pivot the weather dataframe
weather = weather.pivot(index='date', columns='station_name', values=['ev_transpiration','max_temp','min_temp','max_humid','min_humid','solar'])

# Flatten multi-index columns
weather.columns = ['_'.join(col).strip() for col in weather.columns.values]

# Reset index
weather = weather.reset_index()

# Set date column as date type
weather.date = pd.to_datetime(weather.date, format='%d/%m/%Y')

#--------------------------------------------
## Read Eletricity Demand Data - previosly filtered and formatted in open office
# source https://www.spotmarketindex.cz/ 
usage = pd.read_csv(file_path+'daily_avg_electricity_price_2024_filtered.csv', sep=',')
usage.columns = usage.columns.str.lower()
print(usage.head())

## Filter only date and total demand columns
selected_col = ['date','daily_average']
usage = usage[selected_col]

## convert column type
#usage['date'] = pd.to_datetime(usage['date'])
usage['date'] = pd.to_datetime(usage['date'], format='%d/%m/%Y')
usage['daily_average'] = usage['daily_average'].astype(float)
#--------------------------------------------

