import math
import datetime

import pandas as pd
import quandl
import numpy as np

from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
from matplotlib import style

import pickle


# https://matplotlib.org/3.2.1/gallery/style_sheets/style_sheets_reference.html
style.use('ggplot')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

quandl.ApiConfig.api_key = "L1k3HbsN3zssApibtxhz"
data_frame = quandl.get("WIKI/GOOGL")
# print(data_frame.head())

data_frame = data_frame[["Adj. Open", "Adj. High", "Adj. Low", "Adj. Close", "Adj. Volume", ]]

data_frame["Hi Low Percent"] = (data_frame["Adj. High"] - data_frame["Adj. Low"]) / data_frame["Adj. Low"] * 100.0
data_frame["Open Close Percent"] = (data_frame["Adj. Close"] - data_frame["Adj. Open"]) / data_frame["Adj. Open"] * 100.0

data_frame = data_frame[["Adj. Close", "Hi Low Percent", "Open Close Percent", "Adj. Volume"]]
# print(data_frame.head())

forecast_column = "Adj. Close"
data_frame.fillna(-9999, inplace=True)

forecast_shift = int(math.ceil(len(data_frame) * 0.01))
data_frame["Projection"] = data_frame[forecast_column].shift(-forecast_shift)
# print(len(data_frame))
# data_frame.dropna(inplace=True)
# print(len(data_frame))
# print(data_frame.head())
# print(data_frame.tail(50))

features = np.array(data_frame.drop(["Projection"], 1))
features = preprocessing.scale(features)
features = features[:-forecast_shift]     # substituted by data_frame.dropna()

features_recent = features[-forecast_shift:]
# print(len(features))
# print(features)

data_frame.dropna(inplace=True)
# print(len(data_frame))

labels = np.array(data_frame["Projection"])
# print(len(labels))

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2)

# classifier = LinearRegression()
# classifier = svm.SVR()
# classifier.fit(features_train, labels_train)

# with open('linear_regression.pickle', 'wb') as file:
#     pickle.dump(classifier, file)

pickle_file = open('linear_regression.pickle', 'rb')
classifier = pickle.load(pickle_file)

accuracy = classifier.score(features_test, labels_test)
# print(accuracy)

forecast_set = classifier.predict(features_recent)
data_frame["Forecast"] = np.nan
# print(forecast_set)

last_date = data_frame.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    data_frame.loc[next_date] = [np.nan for _ in range(len(data_frame.columns) - 1)] + [i]

data_frame["Adj. Close"].plot()
data_frame["Forecast"].plot()
plt.legend(loc='lower right')
plt.xlabel('Date')
plt.xlabel('Price')
plt.show()
