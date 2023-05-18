"""
Support Vector Regression Model
18 May 2023
Imran Rizki Putranto

Creating a support vector regression model relating an individual's position and their salaries
"""

# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import matplotlib
matplotlib.use('TkAgg')
np.set_printoptions(threshold=sys.maxsize)

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
Y = dataset.iloc[:, -1].values

# Feature Scaling
X = X.reshape((len(X), 1))  # Reshape into 2D array
Y = Y.reshape((len(Y), 1))

sc_x = StandardScaler()  # new values between -3 to 3
X = sc_x.fit_transform(X)

sc_y = StandardScaler()
Y = sc_y.fit_transform(Y)

# Training SVR model on whole dataset
regressor = SVR(kernel='rbf')
regressor.fit(X, Y)

# Predicting new result
val = sc_y.inverse_transform(regressor.predict(sc_x.transform([[6.5]])).reshape(-1, 1))
# Predict and inverse feature scaling

print(val)

# Visualise SVR Result
Y_pred = sc_y.inverse_transform(regressor.predict(X).reshape(-1, 1))
X_inv = sc_x.inverse_transform(X)
Y_inv = sc_y.inverse_transform(Y)

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(X_inv, Y_inv, color='red')
ax1.plot(X_inv, Y_pred, color='blue')
plt.title('Salaries Against Position (SVR)')
plt.xlabel('Position')
plt.ylabel('Salary')

# Visualise SVR Result for higher resolution and smoother curve
fig = plt.figure()
X_grid = np.arange(min(X_inv), max(X_inv), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
Y_pred = sc_y.inverse_transform(regressor.predict(sc_x.transform(X_grid)).reshape(-1, 1))
ax2 = fig.add_subplot(111)
ax2.scatter(X_inv, Y_inv, color='red')
ax2.plot(X_grid, Y_pred, color='blue')
plt.title('Salaries Against Position (SVR)')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()
