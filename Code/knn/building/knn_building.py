import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets
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

# Using 15 principal components
pca = RandomizedPCA(n_components = 15, whiten = True).fit(features_train)
features_train_pca = pca.transform(features_train)
features_test_pca = pca.transform(features_test)

accuracies = list()
neighborss = np.arange(1, 20, 2)
for neighbor in neighborss:
    # we create an instance of Neighbours Classifier and fit the data.
    clf = neighbors.KNeighborsClassifier(n_neighbors = neighbor)
    clf.fit(features_train_pca, labels_train)
    pred = clf.predict(features_test_pca)
    print 'Accuracy (neighbors=', neighbor, ') :',accuracy_score(pred, labels_test) * 100, '%'
    accuracies.append(accuracy_score(pred, labels_test) * 100)
print accuracies
plt.plot(neighborss, accuracies, marker = 'o')
plt.title('Building Prediction - KNN')
plt.xlabel('Neighbors')
plt.ylabel('Accuracies')

plt.show() 
############################################################################

