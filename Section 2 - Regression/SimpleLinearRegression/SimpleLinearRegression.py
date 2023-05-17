"""
Simple Linear Regression Model
17 May 2023
Imran Rizki Putranto

Creating a simple linear regression model relating an individual's years of experience and their salaries
"""

# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from numba import njit
import matplotlib
matplotlib.use('TkAgg')


np.set_printoptions(threshold=sys.maxsize)

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

# Splitting into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Training simple linear regression model onto training set
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Predicting training set results
Y_pred_train = regressor.predict(X_train)

# Predicting test set results
Y_pred_test = regressor.predict(X_test)

# Visualising training set results
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(X_train, Y_train, color='red')
ax1.plot(X_train, Y_pred_train, color='blue')
plt.title('Salaries Against Experiences (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')

# Visualising test set results
fig = plt.figure()
ax2 = fig.add_subplot(111)
ax2.scatter(X_test, Y_test, color='red')
ax2.plot(X_train, Y_pred_train, color='blue')
plt.title('Salaries Against Experiences (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
