import matplotlib.pyplot as plt
from sklearn.svm import SVC

x = [[0.5, 2, 1], [2, 0.5, 1]]
y = [[1, 1, 1], [2, 2, 2]]


plt.scatter(x, y)
plt.xlabel('RSSI')
plt.ylabel('Reference Points')
plt.show()


# Training the data
features = x
label = [1, 2]
clf = SVC()
clf.fit(features, label)
print clf.predict([[1, 2, 1]])