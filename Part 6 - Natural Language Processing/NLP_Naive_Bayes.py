#Natural Language Processing

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

#Cleaning the texts
import re
import nltk
nltk.download('stopwords')#List of irrelevant words
#nltk.download('all')#downloads all nltk data
#nltk.download('popular')#downloads popular data
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
corpus = []#Corpus - Collection of texts
stopwords = stopwords.words('english')
modif_stop = stopwords[:116]+stopwords[119:131]+stopwords[133:143]#Done to remove negative words from irrelevant list
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])#Remove all chars except alphabets
    review = review.lower()#Convert to lower case
    review = review.split()#Split words convert to list
    #Clear irrelevant words and stem remaining words
    review = [ps.stem(word) for word in review if not word in modif_stop]
    #Join to form a string
    review = ' '.join(review)
    corpus.append(review)
    
#Bag of words 
from sklearn.feature_extraction.text import CountVectorizer 
#Max feautres = no.of words to keep(reduces sparcity counts first n common use words)
cv = CountVectorizer(max_features = 1500)#Tokenizing object
X = cv.fit_transform(corpus).toarray()#Sparse matrix(Matrix of features)
y = dataset.iloc[:,1].values


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

'''
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
'''

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Metrices
'''
(TP = # True Positives, TN = # True Negatives,
 FP = # False Positives, FN = # False Negatives):

Accuracy = (TP + TN) / (TP + TN + FP + FN)

Precision = TP / (TP + FP)(Measuring exactness)

Recall = TP / (TP + FN)(Measuring Completeness)

F1 Score = 2 * Precision * Recall / (Precision + Recall)(compromise between Precision and Recall).
'''
TN, FP, FN, TP = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
Accuracy = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1_Score = 2 * Precision * Recall / (Precision + Recall)
print("Accuracy = {0}\nPrecision = {1}\nRecall = {2}\nF1_Score = {3}".format(Accuracy, Precision, Recall, F1_Score))
'''
Naive Bayes result

without negation
Accuracy = 0.73
Precision = 0.6842105263157895
Recall = 0.883495145631068
F1_Score = 0.7711864406779663

with negation
Accuracy = 0.735
Precision = 0.6893939393939394
Recall = 0.883495145631068
F1_Score = 0.7744680851063831
'''

