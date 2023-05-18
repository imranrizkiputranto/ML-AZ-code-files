"""
Backwards Multiple Linear Regression Model
18 May 2023
Imran Rizki Putranto

Creating a multiple linear regression model to determine the best startup to invest

Notes:
    - Feature scaling not needed as coefficients compensate for the scaling
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

np.set_printoptions(threshold=sys.maxsize)

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, 0:-1].values
Y = dataset.iloc[:, -1].values

# Encoding Categorical Data
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')  # Object of column transformer class
X = np.array(ct.fit_transform(X))

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
