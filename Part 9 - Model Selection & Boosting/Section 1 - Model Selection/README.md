# Model Selection and Parameter Tuning Techniques

# K-Fold Cross Validation

Cross-validation, sometimes called rotation estimation or out-of-sample testing, is any of various similar model validation techniques for
assessing how the results of a statistical analysis will generalize to an independent data set. It is mainly used in settings where
the goal is prediction, and one wants to estimate how accurately a predictive model will perform in practice. 
In a prediction problem, a model is usually given a dataset of known data on which training is run (training dataset), 
and a dataset of unknown data (or first seen data) against which the model is tested (called the validation dataset or testing set).
The goal of cross-validation is to test the model's ability to predict new data that was not used in estimating it, in order to flag 
problems like overfitting or selection bias and to give an insight on how the model will generalize to an independent dataset
(i.e., an unknown dataset, for instance from a real problem).

One round of cross-validation involves partitioning a sample of data into complementary subsets,
performing the analysis on one subset (called the training set), and validating the analysis on the other subset
(called the validation set or testing set). 
To reduce variability, in most methods multiple rounds of cross-validation are performed using different partitions,
and the validation results are combined (e.g. averaged) over the rounds to give an estimate of the model's predictive performance.

In summary, cross-validation combines (averages) measures of fitness in prediction to derive a more accurate estimate of 
model prediction performance.


# Grid Search

A model hyperparameter is a characteristic of a model that is external to the model and whose value cannot be estimated from data.
The value of the hyperparameter has to be set before the learning process begins. For example, c in Support Vector Machines, k in k-Nearest 
Neighbors, the number of hidden layers in Neural Networks.

In contrast, a parameter is an internal characteristic of the model and its value can be estimated from data.
Example, beta coefficients of linear/logistic regression or support vectors in Support Vector Machines.


Grid-search is a parameter tuning methodused to find the optimal hyperparameters of a model which results in the most
‘accurate’ predictions.
