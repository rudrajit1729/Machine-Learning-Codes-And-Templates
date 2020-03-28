#Automatic Implementations of Backward Elimination
#Import the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Import the dataset
dataset = pd.read_csv("50_Startups.csv")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

#Encoding the categorical variables
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct=ColumnTransformer([('encoder',OneHotEncoder(),[3])],remainder="passthrough")
X=np.array(ct.fit_transform(X),dtype=np.float)

#Avoiding the dummy variable trap
X=X[:,1:]
#Adding the costant column 
X=np.append(arr=np.ones((50,1)).astype(int),values=X,axis=1)

#Automatic Backward Elimination
import statsmodels.api as sm
#Backward Elimination using p value only
def backwardElimination(x,sl):
    numVars=len(x[0])
    for i in range(0,numVars):
        regressor_OLS=sm.OLS(endog=y,exog=x).fit()
        maxVar=max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0,numVars-i):
                if regressor_OLS.pvalues[j].astype(float)==maxVar:
                    x=np.delete(x,j,1)#arr,index,axis
    regressor_OLS.summary()
    return x

#Backward Elimination with adjusted R squared method
'''Problem with R squared method is R^2 value rises irrespective of the
independent variable being impactful or non impactful whereas adj r^2 value
increases when predictor is significant and decreases when not after each 
inclusion of a new predictor'''
def backwardelimination(x,sl):
    numVars=len(x[0])
    temp=np.zeros((50,6)).astype(int)
    for i in range(0,numVars):
        regressor_OLS=sm.OLS(y,x).fit()
        maxVar=max(regressor_OLS.pvalues).astype(float)
        adjR_before=regressor_OLS.rsquared_adj.astype(float)
        if maxVar > sl:
            for j in range(0,numVars-i):
                if regressor_OLS.pvalues[j].astype(float)==maxVar:
                    temp[:,j]=x[:,j]
                    x=np.delete(x,j,1)
                    tmp_regressor=sm.OLS(y,x).fit()
                    adjR_after=tmp_regressor.rsquared_adj.astype(float)
                    if adjR_before >=adjR_after:
                        #sig pred as adjR^2 decreases after del thus getting to prev model
                        x_rollback=np.hstack((x,temp[:,[0,j]]))
                        #0th col of temp is added as a separator
                        x_rollback=np.delete(x_rollback,j,1)
                        #separator removed
                        #Or
                        #x_rollback=np.hstack((x,temp[:,j])) does the same work
                        print(regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
    regressor_OLS.summary()
    return x

SL=0.05
X_opt=X[:,[0,1,2,3,4,5]]
X_model=backwardelimination(X_opt,SL)

#Splitting the dataset into training and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X_model,y,test_size=0.2,
                                               random_state=0) 

#Fitting the multiple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

#Predicting the test set results
y_pred=regressor.predict(X_test)

y_diff=abs(y_pred-y_test)



        
        
        
        
        
        