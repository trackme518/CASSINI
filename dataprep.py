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
file_path = os.path.join(file_path, 'dataset')
file_path = os.path.join(file_path, '')
print(file_path)

#--------------------------------------------
## Read data
pd.set_option('display.max.columns', 100)

## set the period of forcast in 2021
start_mth = datetime.datetime(2023, 9, 14)
end_mth = datetime.datetime(2024, 9, 14)

## Read Data
dataset = pd.read_csv(file_path+'merged_data_selection.csv', sep=',', encoding='utf-8')
#dataset.columns = dataset.columns.str.lower()
dataset['date'] = pd.to_datetime(dataset['date'], format='%d/%m/%Y')
dataset['electricity_daily_average_CZ'] = dataset['electricity_daily_average_CZ'].astype(float)
print(dataset.head())

#--------------------------------------------

