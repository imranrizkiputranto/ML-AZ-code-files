"""
Polynomial Regression Model
22 May 2023
Imran Rizki Putranto

Creating a polynomial regression model in conjunction with the Combined Cycle Power Plant dataset from the
UCI Machine Learning Repository to predict the energy output based on ambient temperature, exhaust vacuum,
ambient pressure and relative humidity.
"""

# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from numba import njit
import matplotlib
matplotlib.use('TkAgg')
np.set_printoptions(threshold=sys.maxsize)

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

# Splitting into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Training polynomial regression model onto whole set
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X_train)
regressor = LinearRegression()
regressor.fit(X_poly, Y_train)

# Predicting test set results
Y_test_pred = regressor.predict(poly_reg.transform(X_test))
print(np.concatenate((Y_test_pred.reshape(len(Y_test_pred), 1), Y_test.reshape(len(Y_test), 1)), 1))

# Evaluating Model Performance
print(r2_score(Y_test, Y_test_pred))
