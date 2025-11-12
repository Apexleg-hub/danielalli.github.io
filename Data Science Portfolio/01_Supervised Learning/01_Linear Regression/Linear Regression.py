import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Sample data (replace with your actual data)
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)  # Independent variable (feature)
y = np.array([2, 4, 5, 4, 6])  # Dependent variable (target)

# Create a linear regression model
model = LinearRegression()

# Fit the model to the data
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Print model coefficients (intercept and slope)
print("Intercept:", model.intercept_)
print("Coefficient:", model.coef_)

# Plot the data and the regression line
plt.scatter(X, y, color='blue', label='Data')
plt.plot(X, y_pred, color='red', label='Regression Line')
plt.xlabel('Independent Variable (X)')
plt.ylabel('Dependent Variable (y)')
plt.legend()
plt.show()