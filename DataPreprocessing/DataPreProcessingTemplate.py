""" Data PreProcessing Template """

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from numba import njit

np.set_printoptions(threshold=sys.maxsize)

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, 0:-1].values
Y = dataset.iloc[:, -1].values

# Taking care of missing data
# Replace missing salary by average of all salaries

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')  # Creating object for the class and calculating average of salary
imputer.fit(X[:, 1:3])  # find nan values, expects columns with numerical values
X[:, 1:3] = imputer.transform(X[:, 1:3])  # Transform nan values into mean values

# Encoding independent variable
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')  # Object of column transformer class
X = np.array(ct.fit_transform(X))

# Encoding dependent variable
le = LabelEncoder()
Y = np.array(le.fit_transform(Y))

# Splitting into training and test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Feature scaling - Avoid some features to dominate other features, consistent scaling
sc = StandardScaler()  # between -3 to 3
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])


