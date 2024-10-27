import numpy as np
from collections import Counter


class KNN:
    def __init__(self, k = 3):
        self.k = k
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        arr = np.array(y_pred)
        return arr
    def _predict(self, x):
        distance = [np.linalg.norm(x - x_train) for x_train in self.X_train]
        k_indices = np.argsort(distance)[:self.k]
        knearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(knearest_labels).most_common(1)
        return most_common[0][0]
    
    def accuracy(self, y_true, y_pred):
        acc = np.sum(y_true == y_pred)/len(y_true)
        return acc
    

def normalization(X):
    X_min = X.min(axis = 0)
    X_max = X.max(axis = 0)
    scale = (X-X_min)/(X_max - X_min)
    return scale