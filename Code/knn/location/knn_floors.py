import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets
from sklearn.metrics import accuracy_score
from sklearn.decomposition import RandomizedPCA
from sklearn.model_selection import train_test_split



############################################################################
# Importing dataset and getting the features and labels
dataset_train = pd.read_csv("../../trainingData.csv")
dataset_test = pd.read_csv("../../validationData.csv")


floor_train_data =  dataset_train.loc[(dataset_train['BUILDINGID'] == 2) & (dataset_train['FLOOR'] == 2)]

features_train = floor_train_data.iloc[:, 0:520].values
labels_train = floor_train_data.iloc[:, 524].values

ffeatures_train, ffeatures_test, flabels_train, flabels_test = train_test_split(features_train, labels_train, test_size = 0.3, random_state = 42)


# clf_f = neighbors.KNeighborsClassifier(n_neighbors = 2) 
# clf_f.fit(ffeatures_train, flabels_train)
# pred_f = clf_f.predict(ffeatures_test)
# print accuracy_score(pred_f, flabels_test)

# Using 15 principal components
pca_f = RandomizedPCA(n_components = 18, whiten = True).fit(ffeatures_train)
features_train_pca = pca_f.transform(ffeatures_train)
features_test_pca = pca_f.transform(ffeatures_test)

accuracies = list()
neighborss = np.arange(1, 20, 2)
for neighbor in neighborss:
    # we create an instance of Neighbours Classifier and fit the data.
    clf_f = neighbors.KNeighborsClassifier(n_neighbors = neighbor) 
    clf_f.fit(features_train_pca, flabels_train)
    pred_f = clf_f.predict(features_test_pca)
    print 'Accuracy (neighbors=', neighbor, ') :',accuracy_score(pred_f, flabels_test) * 100, '%'
    accuracies.append(accuracy_score(pred_f, flabels_test))
print accuracies
plt.plot(neighborss, accuracies, marker = 'o')
plt.title('Refernce Point Prediction - KNN')
plt.xlabel('Neighbors')
plt.ylabel('Accuracies')

plt.show() 
############################################################################

