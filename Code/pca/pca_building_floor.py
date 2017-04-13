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

# Floors dataset
floors_train = dataset_train.loc[dataset_train['BUILDINGID'] == 1]
floors_test = dataset_test.loc[dataset_test['BUILDINGID'] == 1]

f_features_train = floors_train.iloc[:, 0:520].values
f_labels_train = floors_train.iloc[:, 522].values
f_features_test = floors_test.iloc[:, 0:520].values
f_labels_test = floors_test.iloc[:, 522].values


############################################################################
accuracies_b = list()
accuracies_f = list()
# n_components = np.arange(25,250,25)
n_components = np.arange(5,500,5)

for components in n_components:
	# For Buildings
	pca = RandomizedPCA(n_components = components, whiten = True).fit(features_train)
	features_train_pca = pca.transform(features_train)
	features_test_pca = pca.transform(features_test)
	clf = neighbors.KNeighborsClassifier(n_neighbors = 1) 
	clf.fit(features_train_pca, labels_train)
	pred = clf.predict(features_test_pca)
	print 'Accuracy (n_components=', components, ') :',accuracy_score(pred, labels_test) * 100, '%'
	accuracies_b.append(accuracy_score(pred, labels_test))

	# For floors
	pca_f = RandomizedPCA(n_components = components, whiten = True).fit(f_features_train)
	features_train_pca = pca_f.transform(f_features_train)
	features_test_pca = pca_f.transform(f_features_test)
	clf_f = neighbors.KNeighborsClassifier(n_neighbors = 1) 
	clf_f.fit(features_train_pca, f_labels_train)
	pred_f = clf_f.predict(features_test_pca)
	print 'Accuracy (n_components=', components, ') :',accuracy_score(pred_f, f_labels_test) * 100, '%'
	accuracies_f.append(accuracy_score(pred_f, f_labels_test))

print accuracies_b
f, arr = plt.subplots(2, sharex=True)
arr[0].plot(n_components, accuracies_b, marker='o')
arr[0].set_title('Building Prediction')
arr[1].plot(n_components, accuracies_f, marker='o')
arr[1].set_title('Floor Prediction')
plt.ylabel('Accuracies')
plt.xlabel('Number of features')
plt.show() 


# Conclusion: N = 5 gives the best result for predicting building