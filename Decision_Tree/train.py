from model import DecisionTree
import numpy as np

X = np.random.rand(100,2)
y = np.random.randint(0,2,100)

dt = DecisionTree()
dt.fit(X, y)
y_pred = dt.predict(X)
acc = dt.accuaracy(y, y_pred)
print(f"Accuracy of Decision Tree is: {acc}")
