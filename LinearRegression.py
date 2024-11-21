import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import mean_squared_error, r2_score

iris = load_iris()
X = iris.data[:, 2].reshape(-1, 1)
Y = iris.data[:, 0]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

plt.figure(figsize=(10, 6))
plt.scatter(X_test, Y_test, color='blue', label='Actual Data')
plt.plot(X_test, Y_pred, color='red', linewidth=2, label='Regression Line')
plt.title('Linear Regression: Sepal Length vs Petal Length')
plt.xlabel('Petal Length')
plt.ylabel('Sepal Length')
plt.legend()
plt.show()

residuals = Y_test - Y_pred
plt.figure(figsize=(10, 6))
plt.scatter(Y_pred, residuals, color='green', label='Residuals')
plt.axhline(y=0, color='black', linestyle='--', label='Zero Residual Line')
plt.title('Residuals Plot')
plt.xlabel('Predicted Sepal Length')
plt.ylabel('Residuals')
plt.legend()
plt.show()

mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)
print(f"Mean Squared Error (MSE): {mse}")
print(f"R^2 Score: {r2}")
