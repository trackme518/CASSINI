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

#------------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
#------------------------------------------------
#try Catboost - should ouperform XGBoost
from catboost import Pool, CatBoostRegressor
#------------------------------------------------

import os
file_path = os.path.dirname(os.path.realpath(__file__))
file_path = os.path.join(file_path, 'dataset')
file_path = os.path.join(file_path, '')
print(file_path)

#--------------------------------------------
## Read data
pd.set_option('display.max.columns', 100)

## set the period of forcast
start_mth = datetime.datetime(2023, 7, 31)
end_mth = datetime.datetime(2024, 7, 31)

## Read Data
dataset = pd.read_csv(file_path+'merged_data_big.csv', sep=',', encoding='utf-8')
#dataset.columns = dataset.columns.str.lower()
dataset['date'] = pd.to_datetime(dataset['date'], format='ISO8601')
#dataset['date'] = pd.to_datetime(dataset['date'], format='%d/%m/%Y')
#dataset['electricity_daily_average_CZ'] = dataset['electricity_daily_average_CZ'].astype(float)

## Filter dataset within start and end month
dataset = dataset[( dataset['date']>=start_mth) & (dataset['date'] <= end_mth)]

print(dataset.head())

#--------------------------------------------
"""
## correlation heat map
merge_wo_date = dataset.drop(columns='date')
corr_matrix = merge_wo_date.corr()

##plot the heat map
plt.figure(figsize=(15,8))
sns.heatmap(corr_matrix, annot = True, cmap = 'coolwarm', fmt = ".2f", linewidths=.5, annot_kws={'size':7})
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.title('Correlation Heatmap')
plt.show()

"""
#--------------------------------------------
## Split data into feature and target variable

#dataset.drop(columns=['precipitation_DE', 'precipitation_HU', 'precipitation_RO', 'precipitation_SK', 'electricity_daily_average_CZ', 'wind_speed_CZ','wind_speed_DE','wind_speed_HU','wind_speed_RO','wind_speed_SK'])
#dataset = dataset.drop(columns=['electricity_daily_average_CZ'])
dataset = dataset.drop(columns=['time_series','river_hydropower_AT','river_hydropower_CH','river_hydropower_DE','river_hydropower_ES','river_hydropower_FI','river_hydropower_FR','river_hydropower_IT','river_hydropower_NO','river_hydropower_PT','river_hydropower_RO','river_hydropower_SK'])
X = dataset.drop(columns=['date','price_mwh_eur_DAM']) 
y = dataset['price_mwh_eur_DAM']

#X = dataset.drop(columns=['date','electricity_demand_CZ'])
#y = dataset['electricity_demand_CZ']

## Split data into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

## Set the model dictionary
"""
models = {
'Linear Regression': LinearRegression()
, 'Random Forest Regressor': RandomForestRegressor(random_state = 42)
, 'XGBoost Regressor' : XGBRegressor(random_state = 42)
}
"""

model = XGBRegressor(random_state = 42)

# specify the training parameters
#model = CatBoostRegressor(iterations=15,depth=10,learning_rate=1,loss_function='RMSE')
#model = RandomForestRegressor(random_state = 42)
#model = LinearRegression()

# Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


## Predict and Evaluation
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred) * 100
r_squared = r2_score(y_test, y_pred)

## print model accuracy
print(f"Mean Squared Error = {mse}")
print(f"Mean Absolute Percentage Error = {mape: .2f}%")
print(f"R-squared = {r_squared:.2f}")

## plot scatter, and line comparing test data and prediction
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Actual')
plt.ylabel('Predicted')
p1 = max(max(y_pred), max(y_test))
p2 = min(min(y_pred), min(y_test))
plt.plot([p1, p2], [p1, p2], color = 'r', linestyle = '-', lw=2)
plt.grid(True)
plt.show()
