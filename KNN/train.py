import numpy as np
from model import KNN
from model import normalization

X = np.random.rand(100,2)
y = np.random.randint(0,2,100)
X = normalization(X)

knn = KNN(k = 3)
knn.fit(X, y)
y_pred = knn.predict(X)
knn_acc = knn.accuracy(y_pred, y)
print(f"Accuracy of KNN model is: {knn_acc}")