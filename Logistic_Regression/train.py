import numpy as np
from model import LogisticRegression

X = np.array([[1,4],[6,2],[4,3],[1,4],[1,5]])
y = np.array([0,1,0,0,1])

logi_reg = LogisticRegression()
logi_reg.fit(X, y)
pred = logi_reg.predict(X)
print(f"Predicted values are: {pred}")