from sklearn.naive_bayes import GaussianNB
import pandas as pd 
from time import time

'''---------------------------- Training Section-----------------------------'''
# Reading the training data
dataset_train = pd.read_csv("./fingerprint_train.csv")

# Getting features from the data
# Features: rssi values from 4 different access points
features_train = dataset_train.iloc[:, 2:].values

# Getting labels from the data
# Labels: 4 different refernce points
labels_train = dataset_train.iloc[:, 1].values

clf = GaussianNB()
print 'Starting Training...'
t0 = time()
clf.fit(features_train, labels_train)
print 'Training Time: ', round(time() - t0, 3), 's' 


'''---------------------------- Testing Section-----------------------------'''
dataset_test = pd.read_csv("./fingerprint_test.csv")

# Getting features from the data
# Features: rssi values from 4 different access points
features_test = dataset_test.iloc[:, 2:].values

# Getting labels from the data
# Labels: 4 different refernce points
labels_test = dataset_test.iloc[:, 1].values
print 'Starting Prediction...'
t1 = time()
pred = clf.predict(features_test)
print 'Prediction Time: ', round(time() - t1, 3), 's'
from sklearn.metrics import accuracy_score
print 'Accuracy: ',accuracy_score(pred, labels_test)