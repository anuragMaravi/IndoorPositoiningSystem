import matplotlib.pyplot as plt
from sklearn.svm import SVC

x = [[1, 2.5, 3, 2.5], [2.5, 1, 2.5, 3], [3, 2.5, 1, 2.5], [2.5, 3, 2.5, 1]]
y = [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]]


# plt.scatter(x, y)
# plt.xlabel('RSSI')
# plt.ylabel('Reference Points')
# plt.show()


# Training the data
features = x
labels = [1, 2, 3, 4]
clf = SVC()
clf.fit(features, labels)

# Testing
# Ideal Test
print 'Ideal Test Result:', clf.predict([[2.5, 1, 2.5, 3]])
# Hard Test
print 'Hard Test Result:', clf.predict([[2.75, 3.25, 2.2, 0.8]])