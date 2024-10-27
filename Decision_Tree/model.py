import numpy as np
class Node:
    def __init__(self, feature = None, threshold = None, left = None, right = None,
                 *, value = None):
        """"
        feature: feature index used for spliting
        threshold: threshold values for spliting
        left: left child node
        right: right child node
        value: Class label if it is a leaf node
        """
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        value = self.value is not None
        return value
    
## Create the Decision Tree
class DecisionTree:
    def __init__(self, max_depth = 10, min_sample_split = 2):
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        self.root = None

    def fit(self, X, y):
        # hear Build the Decision Tree
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth = 0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Stoping the condition
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_sample_split):
            leaf_value = self._most_common_label(y)
            node = Node(value=leaf_value)
            return node
        
        # Find the best split
        best_feature, best_threshold = self._best_split(X, y, n_features)

        # If no split is found, create a leaf node
        if best_feature is None:
            leaf_value = self._most_common_label(y)
            node = Node(value = leaf_value)
            return node

        # Split the data and grow the children recurcively
        left_idx, right_idx = self._split(X[:, best_feature], best_threshold)
        left = self._grow_tree(X[left_idx, :], y[left_idx], depth + 1)
        right = self._grow_tree(X[right_idx,:], y[right_idx], depth + 1)
        node = Node(feature=best_feature, threshold=best_threshold, left = left, right = right)
        return node
    
    def _best_split(self, X, y, n_features):
        """Find the best features based on Gini Impurity"""
        best_gini = float("inf")
        best_feature, best_threshold = None, None
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                gini = self._gini_impurity(X[:, feature], y, threshold)
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold
        return best_feature, best_threshold
    

    def _gini_impurity(self, X_column, y, split_threshold):
        """ Calculate the Gini Impurity for a Split at split threshold"""
        left_idx, right_idx = self._split(X_column, split_threshold)
        if len(left_idx) == 0 or len(right_idx) == 0:
            return 0
        
        ## Calculate the Gini Impurity of each Split
        n = len(y)
        n_left, n_right = len(left_idx), len(right_idx)
        gini_left = 1.0 - sum((np.sum(y[left_idx] == c)/n_left)**2 for c in np.unique(y))
        gini_right = 1.0 - sum((np.sum(y[right_idx] == c)/n_right)**2 for c in np.unique(y))

        # Weighted average of the Gini Impurity
        w_avg = (n_left/n)*gini_left + (n_right/n)*gini_right
        return w_avg
    
    def _split(self, X_column, split_threshold):
        """ Split data into Left and Right branches based on the threshold."""
        left_idx = np.argwhere(X_column <= split_threshold).flatten()
        right_idx = np.argwhere(X_column > split_threshold).flatten()
        return left_idx, right_idx
    
    def _most_common_label(self, y):
        """ Return the most commom label in the y(Dependent) variable"""
        counts = np.bincount(y)
        return np.argmax(counts)
    
    def predict(self, X):
        """ Predict class labels for samples in X."""
        pred = np.array([self._traverse_tree(x, self.root) for x in X])
        return pred
    
    def _traverse_tree(self, x, node):
        """ Traverse the Tree to make predictions."""
        if node.is_leaf_node():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
    
    def accuaracy(y_true, y_pred):
        acc = np.sum(y_true == y_pred)/len(y_true)
        return acc
    
