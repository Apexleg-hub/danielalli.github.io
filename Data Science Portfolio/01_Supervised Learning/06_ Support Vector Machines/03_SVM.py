Here are several Python code examples for SVM (Support Vector Machine) implementation:

## 1. Basic SVM with Scikit-learn

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
iris = datasets.load_iris()
X = iris.data[:, :2]  # Using only first two features for visualization
y = iris.target

# Filter to only two classes for binary classification
X = X[y != 2]
y = y[y != 2]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create SVM classifier
svm_classifier = SVC(kernel='linear', C=1.0)

# Train the model
svm_classifier.fit(X_train, y_train)

# Make predictions
y_pred = svm_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

## 2. SVM with Different Kernels


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate sample data
X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, n_informative=2,
                          n_clusters_per_class=1, random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Different kernels to try
kernels = ['linear', 'poly', 'rbf', 'sigmoid']

for kernel in kernels:
    # Create SVM with different kernel
    svm = SVC(kernel=kernel, random_state=42)
    
    # Train the model
    svm.fit(X_train, y_train)
    
    # Make predictions
    y_pred = svm.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Kernel: {kernel}, Accuracy: {accuracy:.3f}")

## 3. SVM for Regression (SVR)


import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Generate sample data for regression
np.random.seed(42)
X = np.sort(5 * np.random.rand(100, 1), axis=0)
y = np.sin(X).ravel() + 0.1 * np.random.randn(100)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create SVR models with different kernels
svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
svr_linear = SVR(kernel='linear', C=100)
svr_poly = SVR(kernel='poly', C=100, degree=3)

# Train the models
svr_rbf.fit(X_train_scaled, y_train)
svr_linear.fit(X_train_scaled, y_train)
svr_poly.fit(X_train_scaled, y_train)

# Make predictions
y_pred_rbf = svr_rbf.predict(X_test_scaled)
y_pred_linear = svr_linear.predict(X_test_scaled)
y_pred_poly = svr_poly.predict(X_test_scaled)

# Evaluate models
print("RBF Kernel - MSE:", mean_squared_error(y_test, y_pred_rbf), "R2:", r2_score(y_test, y_pred_rbf))
print("Linear Kernel - MSE:", mean_squared_error(y_test, y_pred_linear), "R2:", r2_score(y_test, y_pred_linear))
print("Poly Kernel - MSE:", mean_squared_error(y_test, y_pred_poly), "R2:", r2_score(y_test, y_pred_poly))
# Plot results
plt.scatter(X, y, color='red', label='Data')
plt.scatter(X_test, y_pred_rbf, color='blue', label='RBF Predictions')
plt.scatter(X_test, y_pred_linear, color='green', label='Linear Predictions')
plt.scatter(X_test, y_pred_poly, color='orange', label='Poly Predictions')
plt.legend()    
plt.show()

## 4. SVM with Hyperparameter Tuning


import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# Load breast cancer dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define parameter grid for tuning
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf', 'linear', 'poly']
}

# Create GridSearchCV object
grid_search = GridSearchCV(
    SVC(), 
    param_grid, 
    refit=True, 
    cv=5, 
    scoring='accuracy',
    n_jobs=-1
)

# Perform grid search
grid_search.fit(X_train_scaled, y_train)

# Best parameters and score
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)

# Make predictions with best model
best_svm = grid_search.best_estimator_
y_pred = best_svm.predict(X_test_scaled)

# Evaluation
print("\nTest Set Performance:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
```

## 5. Visualizing SVM Decision Boundaries

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.svm import SVC

# Generate sample data
X, y = make_blobs(n_samples=100, centers=2, random_state=42, cluster_std=1.5)

# Create and train SVM
svm = SVC(kernel='linear', C=1.0)
svm.fit(X, y)

# Create mesh to plot decision boundaries
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# Plot decision boundary
Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='black')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('SVM Decision Boundary')
plt.show()
```

## Key Parameters Explained:

- **C**: Regularization parameter (smaller C = softer margin)
- **kernel**: Type of kernel ('linear', 'poly', 'rbf', 'sigmoid')
- **gamma**: Kernel coefficient for 'rbf', 'poly', 'sigmoid'
- **degree**: Degree of polynomial kernel
- **probability**: Whether to enable probability estimates

## Required Libraries:

```bash
pip install scikit-learn matplotlib numpy
```

These examples cover the main aspects of SVM implementation in Python, including classification, regression, hyperparameter tuning, and visualization.