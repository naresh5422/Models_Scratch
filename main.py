import numpy as np

class preprocess:
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def split_data(self, train_ratio = 0.8):
        """
        Split the dataset into train and test sets
        X: independent
        y: Dependent
        X_train(independent): for training the model
        y_train(dependent): for training the model
        X_test(Independent): For validating the model
        y_test(dependent): for validating the model 
        """
        self.train_ratio = train_ratio
        indices = np.arange(self.X.shape[0])
        np.random.shuffle(indices)
        split = int(len(indices)*train_ratio)
        X_train = self.X[indices[:split]]
        X_test = self.X[indices[split:]]
        y_train = self.y[indices[:split]]
        y_test = self.y[indices[split:]]
        return X_train, X_test, y_train, y_test