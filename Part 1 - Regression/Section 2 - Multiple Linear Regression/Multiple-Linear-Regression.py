#Multiple Linear Regression
#Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the dataset
dataset=pd.read_csv("50_Startups.csv")
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

#Encoding the categorical data
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct=ColumnTransformer([('encoder',OneHotEncoder(),[3])],remainder='passthrough')
X=np.array(ct.fit_transform(X),dtype=np.float)

#Avoiding the dummy variable trap(Here it also can be taken care by the libraries implicitly)
X=X[:,1:]

#Splitting the dataset into training and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#Feature Scaling
#Here in multiple regression model it is implicitly taken care of by the libraries

#Fitting Multiple linear regression  to the training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

#Predicting the Test set results
y_pred=regressor.predict(X_test)

#y_diff=abs(y_test-y_pred)

#Building the optimal model using Backward Elimination
import statsmodels.api as sm
X=np.append(arr=np.ones((50,1)).astype(int),values=X,axis=1)
#Significance level=0.05
X_opt=X[:,[0,1,2,3,4,5]]#Optimal independent variables/predictors stored here.
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()#Step2 of backward elimination.
#fitting model with all possible predictors done
regressor_OLS.summary()#Gives summary.find the highest p value
#x2 has highest p value.thus removing it
X_opt=X[:,[0,1,3,4,5]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()
#x1 has highest p value.thus removing it
X_opt=X[:,[0,3,4,5]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()
#x2 has highest p value.thus removing it.i.e. col 4
X_opt=X[:,[0,3,5]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()
#x2 has highest p value.thus removing it i.e. col 5
X_opt=X[:,[0,3]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()
#R&D spend is the optimal independent variable along with a constant

# =============================================================================
#X_train,X_test,y_train,y_test=train_test_split(X_opt,y,test_size=0.2,random_state=0)
#regressor=LinearRegression()
#regressor.fit(X_train,y_train)

##Predicting the Test set results
#y_pred2=regressor.predict(X_test)
#y_diff2=abs(y_test-y_pred2)
# =============================================================================





