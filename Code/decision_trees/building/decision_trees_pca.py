import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.decomposition import RandomizedPCA


############################################################################
# Importing dataset and getting the features and labels
dataset_train = pd.read_csv("../../trainingData.csv")
features_train = dataset_train.iloc[:, 0:520].values
labels_train = dataset_train.iloc[:, 523].values
dataset_test = pd.read_csv("../../validationData.csv")
features_test = dataset_test.iloc[:, 0:520].values
labels_test = dataset_test.iloc[:, 523].values


############################################################################
accuracies = list()
n_components = np.arange(15,505,50)

for components in n_components:
	pca = RandomizedPCA(n_components = components, whiten = True).fit(features_train)
	features_train_pca = pca.transform(features_train)
	features_test_pca = pca.transform(features_test)
	clf = tree.DecisionTreeClassifier()
	clf.fit(features_train_pca, labels_train)
	pred = clf.predict(features_test_pca)
	print 'Accuracy (n_components=', components, ') :',accuracy_score(pred, labels_test) * 100, '%'
	accuracies.append(accuracy_score(pred, labels_test) * 100)

print accuracies
plt.plot(n_components, accuracies, marker = 'o')
plt.title('PCA - Buildings Prediction (Decision Trees)')
plt.xlabel('Features')
plt.ylabel('Accuracies')
plt.show() 