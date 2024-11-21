import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

np.random.seed(42)
X = 2 * np.random.rand(100, 1)  # Random feature values
y = 4 + 3 * X + np.random.randn(100, 1)  # Linear relation with noise (y = 4 + 3x + noise)

alpha = 0.1
iterations = 1000
m = len(X)

theta_0 = 0
theta_1 = 0

cost_history = []

for i in range(iterations):
    h_theta = theta_0 + theta_1 * X
    
    cost = (1 / (2 * m)) * np.sum((h_theta - y) ** 2)
    cost_history.append(cost)
    
    gradient_theta_0 = (1 / m) * np.sum(h_theta - y)
    gradient_theta_1 = (1 / m) * np.sum((h_theta - y) * X)
    
    theta_0 -= alpha * gradient_theta_0
    theta_1 -= alpha * gradient_theta_1

print(f"Gradient Descent results: theta_0 = {theta_0:.4f}, theta_1 = {theta_1:.4f}")

lin_reg = LinearRegression()
lin_reg.fit(X, y)

theta_0_ols = lin_reg.intercept_[0]
theta_1_ols = lin_reg.coef_[0][0]
print(f"OLS (scikit-learn) results: theta_0 = {theta_0_ols:.4f}, theta_1 = {theta_1_ols:.4f}")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(iterations), cost_history)
plt.xlabel('Iterations')
plt.ylabel('Cost (MSE)')
plt.title('Gradient Descent: Cost Function History')

plt.subplot(1, 2, 2)
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X, theta_0 + theta_1 * X, color='red', label='Gradient Descent Regression Line')
plt.plot(X, theta_0_ols + theta_1_ols * X, color='green', label='OLS Regression Line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression Comparison')
plt.legend()

plt.tight_layout()
plt.show()
