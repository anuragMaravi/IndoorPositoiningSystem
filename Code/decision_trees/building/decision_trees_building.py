import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import accuracy_score


############################################################################
# Importing dataset and getting the features and labels
dataset_train = pd.read_csv("../../trainingData.csv")
features_train = dataset_train.iloc[:, 0:520].values
labels_train = dataset_train.iloc[:, 523].values
dataset_test = pd.read_csv("../../validationData.csv")
features_test = dataset_test.iloc[:, 0:520].values
labels_test = dataset_test.iloc[:, 523].values

accuracies = list()
max_depth = np.arange(40, 50, 1)
for depth in max_depth:
    # we create an instance of Neighbours Classifier and fit the data.
    clf = tree.DecisionTreeClassifier(max_depth = 45)
    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)
    print 'Accuracy (depth=', depth ,') :', accuracy_score(pred, labels_test) * 100, '%'
    accuracies.append(accuracy_score(pred, labels_test))
print accuracies
plt.plot(max_depth, accuracies, marker = 'o')
plt.title('Building Prediction - Decision Trees')
plt.xlabel('Max Depth')
plt.ylabel('Accuracies')

plt.show() 
############################################################################
