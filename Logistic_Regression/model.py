import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate = 0.01, epochs = 1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
    
    # Create Sigmoid method
    def sigmoid(self, z):
        sig = 1/(1+np.exp(-z))
        return sig
    
    ## Model Fitting method
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        ## Add Gradiants
        for _ in range(self.epochs):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)
            dw = (1/n_samples)*np.dot(X.T, (y_predicted-y))
            db = (1/n_samples)*np.sum(y_predicted-y)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    ## Create prediction method
    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        results = [1 if i > 0.5 else 0 for i in y_predicted]
        return results