#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 20:18:58 2018

@author: macuser
"""

from pandas import read_csv
#from pandas import datetime
from matplotlib import pyplot

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.dates as mdates
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import datetime

def parser(x):
	#return datetime.strptime('190'+x, '%Y-%m')
    val = datetime.datetime.fromtimestamp(float(x))
    return val.strftime('%Y-%m-%d')
    
#Create Datewise Unique Stock dataset (one record per date)
datastk = pd.read_csv('data_stocks.csv')
datastk["DATEFMT"]=datastk["DATE"].apply(parser)
datastk_uniq = datastk.drop_duplicates(["DATEFMT"])
print(datastk_uniq.head(5))

# SP500 - Auto Correlation Plot
series_sp500=datastk_uniq.iloc[:,1]
autocorrelation_plot(series_sp500)
plt.show()

# Fit ARIMA model (for SP500)
model = ARIMA(series_sp500, order=(5,1,0))
model_fit = model.fit(disp=0)
print(model_fit.summary())
# plot residual errors
residuals = DataFrame(model_fit.resid)
residuals.plot()
pyplot.show()
residuals.plot(kind='kde')
pyplot.show()
print(residuals.describe())

# Predict ARIMA model on test split (for SP500)
X = series_sp500.values
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
	model = ARIMA(history, order=(5,1,0))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()


# Apple Stock Price - Auto Correlation Plot
series_aapl=datastk_uniq.iloc[:,3]
autocorrelation_plot(series_aapl)
plt.show()

# Fit ARIMA model (for AAPL)
model = ARIMA(series_aapl, order=(5,1,0))
model_fit = model.fit(disp=0)
print(model_fit.summary())
# plot residual errors
residuals = DataFrame(model_fit.resid)
residuals.plot()
pyplot.show()
residuals.plot(kind='kde')
pyplot.show()
print(residuals.describe())

# Predict ARIMA model on test split (for Apple Stock)
X = series_aapl.values
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
	model = ARIMA(history, order=(5,1,0))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()
