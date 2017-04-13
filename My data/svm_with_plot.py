from sklearn.svm import SVC
import pandas as pd 
import matplotlib.pyplot as plt


'''---------------------------- Training Section-----------------------------'''
dataset_train = pd.read_csv("./fingerprint_train.csv")
features_train = dataset_train.iloc[:, 2:].values
labels_train = dataset_train.iloc[:, 1].values
dataset_test = pd.read_csv("./fingerprint_test.csv")
features_test = dataset_test.iloc[:, 2:].values
labels_test = dataset_test.iloc[:, 1].values

accuracies = list()
kernels = ['rbf', 'sigmoid','poly', 'linear']
kernels_val = [1, 2, 3, 4]
print 'Accuracies for different kernels:'
for kernel in kernels:
	clf = SVC(kernel = kernel, C = 1.)
	clf.fit(features_train, labels_train)
	pred = clf.predict(features_test)
	from sklearn.metrics import accuracy_score
	accuracies.append(accuracy_score(pred, labels_test))
	print 'Accuracy(', kernel, '): ',accuracy_score(pred, labels_test)
print accuracies
plt.plot(kernels_val, accuracies)
plt.xlabel('Kernels')
plt.ylabel('Accuracies')

plt.show()
