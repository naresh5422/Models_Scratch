import numpy as np

class LinearRegression:
    def __init__(self, learning_rate = 0.01, epochs = 1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
    def fit(self, X, y):
        ## Initialize parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradiant Descent
        for _ in range(self.epochs):
            y_predicted = np.dot(X, self.weights)+self.bias
            
            # Calcute the Gradiants
            dw = (1/n_samples)*np.dot(X.T, (y_predicted-y))
            db = (1/n_samples)*np.sum(y_predicted - y)

            ## Update the weights
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        res = np.dot(X, self.weights)+self.bias
        return res