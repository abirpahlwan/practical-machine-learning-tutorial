import numpy as np
import pandas as pd
from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

data_frame = pd.read_csv('breast-cancer-wisconsin.data')
print(data_frame.head())
data_frame.replace('?', -99999, inplace=True)
data_frame.drop(['id'], 1, inplace=True)

x = np.array(data_frame.drop(['class'], 1))
y = np.array(data_frame['class'])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = neighbors.KNeighborsClassifier()
model.fit(x_train, y_train)

accuracy = model.score(x_test, y_test)
print("Accuracy:", accuracy)
