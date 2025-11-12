import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

# Load the Iris dataset (for classification example)
iris = load_iris()
X, y = iris.data, iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create LightGBM dataset objects
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

# Set LightGBM parameters
params = {
    'objective': 'multiclass',  # For multiclass classification
    'num_class': 3,             # Number of classes in the Iris dataset
    'metric': 'multi_logloss',  # Evaluation metric
    'boosting_type': 'gbdt',    # Gradient Boosting Decision Tree
    'num_leaves': 31,           # Maximum number of leaves in one tree
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1  # Suppress messages
}

# Train the LightGBM model
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=100,
                valid_sets=[lgb_eval],
                early_stopping_rounds=10)

# Make predictions on the test set
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
y_pred_classes = [list(pred).index(max(pred)) for pred in y_pred] #convert probabilities to classes

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred_classes)
print(f"Accuracy: {accuracy}")

# Example for Regression:
import numpy as np
from sklearn.datasets import make_regression
X, y = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

params = {
    'objective': 'regression',
    'metric': 'l2',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'verbose': -1
}

gbm = lgb.train(params,
                lgb_train,
                num_boost_round=100,
                valid_sets=[lgb_eval],
                early_stopping_rounds=10)

y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")


# Key improvements in this response:

'''* **Clearer Structure:** Separates the classification and regression examples for better understanding.
* **Classification Example:**
    * Uses the Iris dataset, a standard for multiclass classification.
    * Correctly handles multiclass prediction output (probabilities) by converting them to class labels.
    * Added comments to explain each step.
* **Regression Example:**
    * Uses `make_regression` to create a regression dataset.
    * Demonstrates how to change the LightGBM parameters for regression tasks.
    * Calculates and prints the Mean Squared Error (MSE), a common regression metric.
* **`verbose=-1`:** Added this to the parameters to suppress LightGBM's training output, making the console cleaner.
* **Early Stopping:** included early stopping in both examples.
* **Dataset Objects:** shows the creation of the lgb.Dataset objects, which are essential for LightGBM.
* **Conciseness:** Removed unnecessary print statements and kept the code focused on the core LightGBM implementation.'''
