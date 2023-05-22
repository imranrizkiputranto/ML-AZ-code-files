"""
Backwards Multiple Linear Regression Model
22 May 2023
Imran Rizki Putranto

Creating a polynomial regression model in conjunction with the Combined Cycle Power Plant dataset from the
UCI Machine Learning Repository to predict the energy output based on ambient temperature, exhaust vacuum,
ambient pressure and relative humidity.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use('TkAgg')
np.set_printoptions(threshold=sys.maxsize)

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

# Splitting into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Training multiple linear regression model on training set
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Predicting training set results
Y_pred_train = regressor.predict(X_train)
Y_pred_train = Y_pred_train.reshape((len(Y_pred_train), 1))

# Predicting test set results
Y_pred_test = regressor.predict(X_test)
Y_pred_test = Y_pred_test.reshape((len(Y_pred_test), 1))

# Comparing predicted and actual results
Y_test = Y_test.reshape((len(Y_test), 1))
print(np.concatenate((Y_pred_test, Y_test), 1))

# Evaluating model performance
print(r2_score(Y_test, Y_pred_test))
