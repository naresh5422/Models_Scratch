import numpy as np
from model import LinearRegression

# Generate some data points
X = np.array([[1,1], [2,2], [3,3], [4,4]])
y = np.array([6,8,10,12])

# Initialize and train the model
model = LinearRegression(learning_rate = 0.01, epochs = 1000)
model.fit(X, y)
pred = model.predict(X)
print(f"Predictions are: {pred}")

