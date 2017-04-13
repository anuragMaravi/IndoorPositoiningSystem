import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.svm import SVC
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

# Using 35 principal components
pca = RandomizedPCA(n_components = 35, whiten = True).fit(features_train)
features_train_pca = pca.transform(features_train)
features_test_pca = pca.transform(features_test)

accuracies = list()
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
kernel_val = np.arange(0, len(kernels))
for kernel in kernels:
    # we create an instance of Neighbours Classifier and fit the data.

    clf = SVC(kernel = kernel, C = 1.)
    clf.fit(features_train_pca, labels_train)
    pred = clf.predict(features_test_pca)
    print 'Accuracy (kernels=', kernel ,') :',accuracy_score(pred, labels_test) * 100, '%'
    accuracies.append(accuracy_score(pred, labels_test))
print accuracies
plt.plot(kernel_val, accuracies, marker = 'o')
plt.title('Building Prediction - SVM')
plt.xticks(kernel_val, kernels)
plt.xlabel('Kernels')
plt.ylabel('Accuracies')

plt.show() 
############################################################################

