"""
Polynomial Regression Model
18 May 2023
Imran Rizki Putranto

Creating a polynomial regression model relating an individual's position and their salaries
"""

# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from numba import njit
import matplotlib
matplotlib.use('TkAgg')
np.set_printoptions(threshold=sys.maxsize)

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
Y = dataset.iloc[:, -1].values

# Training linear regression model onto whole set
lin_reg = LinearRegression()
lin_reg.fit(X, Y)

# Training polynomial regression model onto whole set
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, Y)

# Visualising linear regression results
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(X, Y, color='red')
ax1.plot(X, lin_reg.predict(X), color='blue')
plt.title('Salary Against Position (Linear Model)')
plt.xlabel('Position')
plt.ylabel('Salary')

# Visualising polynomial regression results
fig = plt.figure()
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
ax2 = fig.add_subplot(111)
ax2.scatter(X, Y, color='red')
# ax2.plot(X, lin_reg_2.predict(X_poly), color='blue')
ax2.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color='blue')
plt.title('Salary Against Position (Polynomial Model')
plt.xlabel('Position')
plt.ylabel('Salary')
# plt.show()

# Predicting result with linear regression
print(lin_reg.predict([[6.5]]))

# Predicting result with polynomial regression
print(lin_reg_2.predict(poly_reg.fit_transform([[6.5]])))
