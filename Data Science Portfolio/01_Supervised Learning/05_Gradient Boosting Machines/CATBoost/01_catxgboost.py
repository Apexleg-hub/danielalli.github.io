#**CatBoost** (Categorical Boosting) is a gradient-boosting library designed to handle categorical features natively and efficiently. 
# It is particularly useful for datasets with categorical variables and requires minimal preprocessing. Here's a guide to using CatBoost in Python:

#---

### **1. Installation**

#!pip install catboost


#---

### **2. Basic Regression Example**

from catboost import CatBoostRegressor, Pool
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load dataset
data = fetch_california_housing()
X, y = data.data, data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize model
model = CatBoostRegressor(
    iterations=1000,          # Number of trees
    learning_rate=0.1,
    depth=6,                  # Tree depth
    loss_function='RMSE',     # Regression metric
    eval_metric='RMSE',
    early_stopping_rounds=10,
    verbose=False             # Disable training logs
)

# Train model
model.fit(
    X_train, y_train,
    eval_set=(X_test, y_test),
    cat_features=[]           # Specify categorical features (if any)
)

# Predictions
y_pred = model.predict(X_test)
print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False):.4f}")


#---

### **3. Classification Example (with Categorical Features)**

from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score

# Example dataset with categorical features
import pandas as pd
data = pd.DataFrame({
    'Age': [25, 30, 35, 40],
    'City': ['NY', 'SF', 'NY', 'LA'],  # Categorical column
    'Salary': [70_000, 90_000, 85_000, 120_000],
    'Target': [0, 1, 0, 1]
})

X = data.drop('Target', axis=1)
y = data['Target']

# Identify categorical features
cat_features = ['City']  # Explicitly declare categorical columns

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize classifier
model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=4,
    loss_function='Logloss',  # Binary classification
    eval_metric='Accuracy',
    cat_features=cat_features,  # Automatically handles encoding
    verbose=False
)

# Train model
model.fit(X_train, y_train, eval_set=(X_test, y_test))

# Predictions
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")


#---

### **4. Key Features**

#### **Automatic Categorical Handling**
#- No need for manual encoding (one-hot, label).
#- Specify categorical columns using `cat_features`:

model.fit(X, y, cat_features=[0, 2])  # Columns 0 and 2 are categorical
  

#### **GPU Acceleration**

model = CatBoostClassifier(task_type='GPU', devices='0:1')  # Use GPU


#### **Missing Value Handling**
#- Automatically handles missing values without imputation.

#### **Feature Importance**

model.get_feature_importance(prettified=True)


#---

### **5. Advanced Usage**

#### **Cross-Validation**

from catboost import cv

params = {
    'loss_function': 'Logloss',
    'iterations': 1000,
    'learning_rate': 0.1,
    'depth': 6
}

cv_data = cv(
    Pool(X, y, cat_features=cat_features),
    params=params,
    fold_count=5,
    verbose=False
)


#### **Hyperparameter Tuning**

grid = {
    'depth': [4, 6, 8],
    'learning_rate': [0.01, 0.1, 0.2],
    'l2_leaf_reg': [1, 3, 5]
}

model.grid_search(grid, X_train, y_train, cv=3)


#### **Visualization**

from catboost import MetricVisualizer

# Track training metrics
MetricVisualizer(['catboost_training.log']).start()


#---

### **6. Advantages of CatBoost**
'''1. **Categorical Feature Support**: No manual preprocessing.
2. **Robustness**: Reduces overfitting with ordered boosting.
3. **Speed**: Optimized for GPU training.
4. **Interpretability**: Built-in SHAP values and feature importance.

#---

### **7. When to Use CatBoost**
- Datasets with **categorical features**.
- Competitions like Kaggle where accuracy matters.
- Situations requiring minimal preprocessing.'''

#---

### **8. Full Workflow Example**

# Load data with categorical features
train_data = Pool(
    data=X_train,
    label=y_train,
    cat_features=['City', 'Gender']  # Example columns
)

# Train model
model = CatBoostClassifier(iterations=1000)
model.fit(train_data, plot=True)  # Real-time training visualization

# Save model
model.save_model('catboost_model.cbm')

# Load model
loaded_model = CatBoostClassifier()
loaded_model.load_model('catboost_model.cbm')


'''CatBoost simplifies working with categorical data while maintaining high performance. 
Its native handling of categorical features often makes it a better choice than XGBoost 
or LightGBM for datasets with many categorical variables.'''