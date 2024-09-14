import pandas as pd
import datetime
# load the model from disk
import pickle

import os

from xgboost import XGBRegressor

file_path = os.path.dirname(os.path.realpath(__file__))
file_path = os.path.join(file_path, 'dataset')
file_path = os.path.join(file_path, '')
print(file_path)

dataset = pd.read_csv(file_path+'merged_data_big.csv', sep=',', encoding='utf-8')
dataset['date'] = pd.to_datetime(dataset['date'], format='ISO8601')


## set the period for the forcast
start_mth = datetime.datetime(2023, 5, 1)
end_mth = datetime.datetime(2023, 5, 30)


## Filter dataset within start and end month
dataset = dataset[( dataset['date']>=start_mth) & (dataset['date'] <= end_mth)]
dataset_test_truth = dataset['price_mwh_eur_DAM']
dataset_test = dataset.drop(columns=['date', 'price_mwh_eur_DAM'])


file_path_model = os.path.dirname(os.path.realpath(__file__))
file_path_model = os.path.join(file_path_model, 'finalized_model.sav')
print(file_path_model)


# load the model from disk
model = pickle.load(open(file_path_model, 'rb'))
# expects 146 dimensions

ynew = model.predict(dataset_test)

# show the inputs and predicted probabilities
for idx, x in enumerate(dataset_test_truth):
    print("X=%s, Predicted=%s" % (x, ynew[idx]))

result = model.score(dataset_test, dataset_test_truth)
print(result)