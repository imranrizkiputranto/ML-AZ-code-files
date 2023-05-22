"""
Support Vector Regression Model
22 May 2023
Imran Rizki Putranto

Creating a support vector regression model in conjunction with the Combined Cycle Power Plant dataset from the
UCI Machine Learning Repository to predict the energy output based on ambient temperature, exhaust vacuum,
ambient pressure and relative humidity.
"""

# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use('TkAgg')
np.set_printoptions(threshold=sys.maxsize)

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

# X = X.reshape((len(X), 1))  # Reshape into 2D array
Y = Y.reshape((len(Y), 1))

# Splitting into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Feature Scaling
sc_x = StandardScaler()  # new values between -3 to 3
X_train = sc_x.fit_transform(X_train)

sc_y = StandardScaler()
Y_train = sc_y.fit_transform(Y_train)

# Training SVR model on whole dataset
regressor = SVR(kernel='rbf')
regressor.fit(X_train, Y_train)

# Predicting new result
Y_test_pred = sc_y.inverse_transform(regressor.predict(sc_x.transform(X_test)).reshape(-1, 1))
# Predict and inverse feature scaling

print(np.concatenate((Y_test_pred.reshape(len(Y_test_pred), 1), Y_test.reshape(len(Y_test), 1)), 1))

# Evaluating Model Performance
print(r2_score(Y_test, Y_test_pred))
