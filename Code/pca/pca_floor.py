import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets
from sklearn.metrics import accuracy_score
from sklearn.decomposition import RandomizedPCA


############################################################################
# Importing dataset and getting the features and labels
dataset_train = pd.read_csv("../trainingData.csv")
features_train = dataset_train.iloc[:, 0:520].values
labels_train = dataset_train.iloc[:, 523].values
dataset_test = pd.read_csv("../validationData.csv")
features_test = dataset_test.iloc[:, 0:520].values
labels_test = dataset_test.iloc[:, 523].values

floors_train = dataset_train.loc[dataset_train['BUILDINGID'] == 1]
floors_test = dataset_test.loc[dataset_test['BUILDINGID'] == 1]

f_features_train = floors_train.iloc[:, 0:520].values
f_labels_train = floors_train.iloc[:, 522].values
f_features_test = floors_test.iloc[:, 0:520].values
f_labels_test = floors_test.iloc[:, 522].values

############################################################################
# PCA Code starts here
############################################################################
accuracies = list()
n_components = [5, 10, 15, 20, 25]
for components in n_components:
	pca = RandomizedPCA(n_components = components, whiten = True).fit(f_features_train)
	features_train_pca = pca.transform(f_features_train)
	features_test_pca = pca.transform(f_features_test)
	clf = neighbors.KNeighborsClassifier(n_neighbors = 1) 
	clf.fit(features_train_pca, f_labels_train)
	pred = clf.predict(features_test_pca)
	print 'Accuracy (n_components=', components, ') :',accuracy_score(pred, f_labels_test) * 100, '%'
	accuracies.append(accuracy_score(pred, f_labels_test) * 100)

print accuracies
plt.plot(n_components, accuracies)
plt.title('n Components vs Accuracies')
plt.xlabel('n Components')
plt.ylabel('Accuracies')

plt.show() 


# Conclusion: N = 5 gives the best result for predicting building