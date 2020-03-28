#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset=pd.read_csv("Data.csv")
X=dataset.iloc[:,:-1].values#Matrix of features
y=dataset.iloc[:,-1].values#Dependent variable vector

#Handling missing data
from sklearn.impute import SimpleImputer
missingvalues=SimpleImputer(missing_values=np.nan,strategy="mean",verbose=0)
missingvalues=missingvalues.fit(X[:,1:3])
X[:,1:3]=missingvalues.transform(X[:,1:3])

#Encoding categorical data
#encoding the independent variable
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct=ColumnTransformer([('encoder',OneHotEncoder(),[0])],remainder="passthrough")
X=np.array(ct.fit_transform(X),dtype=np.float)
#encoding the dependent variable
from sklearn.preprocessing import LabelEncoder
y=LabelEncoder().fit_transform(y)

#Splitting the dataset into train and test sets
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
'''(arrays,test_size,seed to the random generator:if not given sets are not
reproducible)'''

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)
#Fit the obj in training set and then transform it.For test set we only transform
#Scaling dummy variables depends upon context.Here we do it
#to avoid fit and transform X_train[:,3:]
#Scaling dependent variable not needed here.
#for regression models we apply so as huge range of values
sc_y=StandardScaler()
y_train=sc_y.fit_transform(y_train.reshape(-1,1))
y_test=sc_y.transform(y_test.reshape(-1,1))
#Passing 1d array deprecated thus reshape used to reshape array.-1 signifies unknown dimension.



