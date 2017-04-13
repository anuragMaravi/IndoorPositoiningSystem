import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets
from sklearn.metrics import accuracy_score
from sklearn.decomposition import RandomizedPCA


############################################################################
# Importing dataset and getting the features and labels
dataset_train = pd.read_csv("../../trainingData.csv")
dataset_test = pd.read_csv("../../validationData.csv")

# Floors dataset
floors_train = dataset_train.loc[dataset_train['BUILDINGID'] == 2]
floors_test = dataset_test.loc[dataset_test['BUILDINGID'] == 2]

f_features_train = floors_train.iloc[:, 0:520].values
f_labels_train = floors_train.iloc[:, 522].values
f_features_test = floors_test.iloc[:, 0:520].values
f_labels_test = floors_test.iloc[:, 522].values

# Using 15 principal components
pca_f = RandomizedPCA(n_components = 15, whiten = True).fit(f_features_train)
features_train_pca = pca_f.transform(f_features_train)
features_test_pca = pca_f.transform(f_features_test)

accuracies = list()
neighborss = np.arange(1, 20, 2)
for neighbor in neighborss:
    # we create an instance of Neighbours Classifier and fit the data.
    clf_f = neighbors.KNeighborsClassifier(n_neighbors = neighbor) 
    clf_f.fit(features_train_pca, f_labels_train)
    pred_f = clf_f.predict(features_test_pca)
    print 'Accuracy (neighbors=', neighbor, ') :',accuracy_score(pred_f, f_labels_test) * 100, '%'
    accuracies.append(accuracy_score(pred_f, f_labels_test))
print accuracies
plt.plot(neighborss, accuracies, marker = 'o')
plt.title('Floors Prediction - KNN')
plt.xlabel('Neighbors')
plt.ylabel('Accuracies')

plt.show() 
############################################################################

