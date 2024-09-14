# CASSINI
CASSINI Hackthaton challange #2 - sustainable living.

## Goal
Predict electricity prices based on weather and other data from sattelites using machine learning AI model.

## Data sources

* Solar radiation
    * https://ads.atmosphere.copernicus.eu/cdsapp#!/dataset/cams-solar-radiation-timeseries?tab=form
    * Altitutde -999., universal time, 50.74424, 9.84271 - Germany, 49.94270, 15.11549 - Czechia
* Electricity price data
    * https://www.spotmarketindex.cz/
    * 1/1/2019-14/9/2024 daily averages
* Merged data
    * https://cds-beta.climate.copernicus.eu/datasets/sis-energy-derived-projections?tab=download

## Observations
We can fairly accuretly predict electricity demand with the XGBoost Regressor - improved by scaler:
Mean Squared Error = 37873863.09106111
Mean Absolute Percentage Error =  0.16%
R-squared = 0.98

However it fails when trying to preditct the electricity prices taken from www.spotmarketindex.cz. Prices were taken outside the original dataset from spot market values. The Cassini dataset might be inaccurate or the spot prices rely on more variables then just the weather data - highly likely. When confiding the dataset to last year we get these results:

XGBoost Regressor
Mean Squared Error = 325364.434709683
Mean Absolute Percentage Error =  54.00%
R-squared = 0.45

That is not accepteble. However when trying with DAM - day ahead market price (https://peaksubstation.com/day-ahead-vs-real-time-energy-markets/) index prices repoted by https://www.ote-cr.cz/en/statistics/yearly-market-report?date=2024-01-01 it seems better:

Mean Squared Error = 4617.042284111203
Mean Absolute Percentage Error =  24.72%
R-squared = -0.04

We have also tried "Random forest" algorithm (Mean Absolute Percentage Error =  25.99%) instead of "XGBRegressor" (Mean Absolute Percentage Error =  25.07%) and Linear regression (Mean Absolute Percentage Error =  28.76%).

Still not good enough thought. Maybe we need to think more about the data.
![variables correlation heatmap graph](dataset/correlationVariableMap_v2.png)

From the variables correlation heatmap graph we can see some interesting relations. For example the electricity demand in Czech Republic has less impact then electricity demand in Deutschland and suprisingly in Slovakia as well. On the other hand we can see that wind speed, precipitation and on shore hydro energy power output is not much of a factor.

By dropping these variables we improve the result slightly:

Mean Squared Error = 433.7925148901215
Mean Absolute Percentage Error =  24.59%
R-squared = 0.46

By restricting the data range to 1 year only we also get slightly better result:

Mean Squared Error = 323.5808738722185
Mean Absolute Percentage Error =  21.38%
R-squared = 0.60





