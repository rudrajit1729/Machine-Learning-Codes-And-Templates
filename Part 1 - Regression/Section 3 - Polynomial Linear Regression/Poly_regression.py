#Polynomial Regression

#Import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Import the dataset
dataset=pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

#plt.plot(X,y,color='red')
#We didnt split the dataset as it is very small and for accuracy we take entire set
#Feautre scaling automatically handled in regression models

#Fitting Linear regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X,y)

#Fitting Polynomial regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly,y)

#Visualising Linear regression results
plt.scatter(X,y,color='red')
plt.plot(X,lin_reg.predict(X),color='blue')
plt.title("Truth or Bluff(Linear Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

#Visualising Polynomial regression results
X_grid=np.arange(min(X),max(X),0.01)#Smoothing curve by taking one smallest division as 0.01
X_grid=X_grid.reshape(len(X_grid),1)#converting vector into matrix of features
plt.scatter(X,y,color='red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)),color='blue')
plt.title("Truth or Bluff(Linear Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

#Predicting a new result with Linear Regression
#Level 6.5

x=np.array(float(input("Enter position level:")))
lin_reg.predict(x.reshape(-1,1))

#Predicting a new result with Polynomial Regression
x=np.array(float(input("Enter position level:")))
lin_reg2.predict(poly_reg.fit_transform(x.reshape(-1,1)))#Level 6.5



