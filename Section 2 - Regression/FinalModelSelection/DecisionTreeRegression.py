"""
Decision Tree Regression Model
22 May 2023
Imran Rizki Putranto

Creating a decision tree regression model in conjunction with the Combined Cycle Power Plant dataset from the
UCI Machine Learning Repository to predict the energy output based on ambient temperature, exhaust vacuum,
ambient pressure and relative humidity.
"""

# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
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

# Training decision tree regression model on training set
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X_train, Y_train)

# Predicting test set results
Y_test_pred = regressor.predict(X_test)
print(np.concatenate((Y_test_pred.reshape(len(Y_test_pred), 1), Y_test.reshape(len(Y_test), 1)), 1))

# Evaluating model performance
print(r2_score(Y_test, Y_test_pred))
