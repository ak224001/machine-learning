# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 00:03:35 2019

@author: Aditya
"""
"""To pridict the truth and Bluff of an employee's salary....he is saying his salary was 160000$...
First will create linear regression model then Polynomail regression model"""


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#Fitting linear regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X,Y)

#Fitting polynomail regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
Poly_reg= PolynomialFeatures(degree=4)
X_poly = Poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,Y)

#Visualising the linear regression result
plt.scatter(X,Y,color="red")
plt.plot(X,lin_reg.predict(X),color="blue")
plt.title("Truth or Bluff(Linear Regression)")
plt.xlabel("Positive level")
plt.ylabel("salary")
plt.show();

#Visualising the Polynomail regression result
plt.scatter(X,Y,color="red")
plt.plot(X,lin_reg_2.predict(Poly_reg.fit_transform(X)),color="blue")
plt.title("Truth or Bluff(Polynomail Regression)")
plt.xlabel("Positive level")
plt.ylabel("salary")
plt.show();

#predicting a new result with linear regression
lin_reg.predict(6.50)
"""Linear regression model predicting salary is 330378$...So this model is not Goood"""

#predicting a new result with Polynomail regression
lin_reg_2.predict(Poly_reg.fit_transform(6.5))

"""Polynomial regression model predicting salary is 160000$...So this model is Perfect"""