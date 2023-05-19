"""
Decision Tree Regression Model
19 May 2023
Imran Rizki Putranto

Creating a decision tree regression model relating an individual's position and their salaries
"""

# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
import matplotlib
matplotlib.use('TkAgg')
np.set_printoptions(threshold=sys.maxsize)

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
Y = dataset.iloc[:, -1].values

# Training decision tree regression model on whole dataset
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, Y)

# Predicting new result
print(regressor.predict([[6.5]]))

# Visualising decision tree regression results
fig = plt.figure()
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
Y_pred = regressor.predict(X_grid).reshape(-1, 1)
ax2 = fig.add_subplot(111)
ax2.scatter(X, Y, color='red')
ax2.plot(X_grid, Y_pred, color='blue')
plt.title('Salaries Against Position (Decision Tree)')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()


