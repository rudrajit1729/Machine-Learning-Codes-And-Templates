# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 20:56:19 2019

@author: RUDRAJIT
"""

# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:,3:13].values
y = dataset.iloc[:, 13].values

#Encoding the categorical variables
#Encoding the gender column
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:, 2] = labelencoder_X.fit_transform(X[:, 2])
#Encoding the country column
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct=ColumnTransformer([('encoder',OneHotEncoder(),[1])],remainder="passthrough")
X=np.array(ct.fit_transform(X),dtype=np.float)
#Avoiding the dummy variable trap
X = X[:, 1:]

#Splitting the dataset into train and test sets
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#Part 2- Making the ANN

#Importing the keras library and packages
from tensorflow import keras 
#import keras
from keras.models import Sequential #Initializes the NN
from keras.layers import Dense #Creates the layers of the ANN
from keras.models import load_model #To save and load model

# fix random seed for reproducibility
seed = 42
np.random.seed(seed)

try:
    #Loading the model
    classifier = load_model('model_file.h5')
    print("Model Loaded Successfully...")
    
except:
    #Initialising the ANN(Defining it as a sequence of layers)
    classifier = Sequential()
    
    #Adding the input layer and the first hidden layer
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    
    #Adding the second hidden layer
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    
    #Adding the output layer
    classifier.add(Dense(units = 1, kernel_initializer= 'uniform', activation = 'sigmoid'))
    
    #Compiling the ANN(Applying Stochastic Gradient Descent on whole ANN)
    classifier.compile(optimizer ='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    #Fitting the ANN to the training set
    classifier.fit(X_train, y_train, batch_size= 10, epochs = 100)
    
    # Save the model
    classifier.save('model_file.h5')
    print("Model Created and Saved Successfully...")

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Evaluate the model
scores = classifier.evaluate(X_test, y_test)
print("\n%s: %.2f%%" % (classifier.metrics_names[1], scores[1]*100))
#accuracy: 85.95%

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


#Here is the code for future reusability of the model
'''
import numpy as np 
from keras.models import load_model # To save and load model

# Load the model
classifier = load_model('model_file.h5')

# Load the test data file and make predictions on it
predictions = classifier.predict(np.loadtxt("modiftest.csv", delimiter=";"))

print(predictions.shape)

my_predictions=classifier.predict(predictions)

print(my_predictions)
'''