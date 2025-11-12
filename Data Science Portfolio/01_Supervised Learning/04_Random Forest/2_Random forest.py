# I'll provide you with a comprehensive guide to Random Forest, including code examples and the types of data you can use.

## Random Forest Implementation

### 1. Basic Random Forest Classification


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns

# Load sample dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train Random Forest classifier
rf_classifier = RandomForestClassifier(
    n_estimators=100,      # Number of trees
    max_depth=5,           # Maximum depth of trees
    min_samples_split=2,   # Minimum samples required to split a node
    min_samples_leaf=1,    # Minimum samples required at a leaf node
    random_state=42
)

rf_classifier.fit(X_train, y_train)

# Make predictions
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': iris.feature_names,
    'importance': rf_classifier.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)
```

### 2. Random Forest Regression

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_boston
import numpy as np

# Load regression dataset
boston = load_boston()
X = boston.data
y = boston.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train Random Forest regressor
rf_regressor = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

rf_regressor.fit(X_train, y_train)

# Make predictions
y_pred = rf_regressor.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")


### 3. Advanced Random Forest with Hyperparameter Tuning


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Create Random Forest classifier
rf = RandomForestClassifier(random_state=42)

# Perform grid search
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    scoring='accuracy'
)

grid_search.fit(X_train, y_train)

# Best parameters and score
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-validation Score:", grid_search.best_score_)

# Use best model for predictions
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.2f}")


### 4. Visualization of Results


# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance, x='importance', y='feature')
plt.title('Feature Importance in Random Forest')
plt.tight_layout()
plt.show()

# Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()


## Types of Data You Can Use with Random Forest

### 1. **Structured Data (Most Common)**
- **Tabular data**: CSV files, Excel spreadsheets, databases
- **Numerical features**: Age, salary, temperature, price
- **Categorical features**: Gender, country, product category


# Example: Customer churn prediction
customer_data = pd.DataFrame({
    'age': [25, 45, 35, 50, 28],
    'salary': [50000, 80000, 60000, 90000, 45000],
    'tenure': [2, 5, 3, 7, 1],
    'gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
    'churn': [0, 1, 0, 1, 0]  # Target variable
})
```

### 2. **Classification Problems**
- **Binary classification**: Spam detection, fraud detection
- **Multi-class classification**: Image recognition, sentiment analysis
- **Multi-label classification**: Document categorization

### 3. **Regression Problems**
- **Continuous values**: House prices, stock prices, temperature
- **Time series forecasting**: Sales prediction, weather forecasting

### 4. **Specific Data Types**

#### Numerical Data

# Perfect for Random Forest
numerical_features = ['age', 'income', 'height', 'weight']


#### Categorical Data

# Need encoding
from sklearn.preprocessing import LabelEncoder

categorical_data = ['red', 'blue', 'green', 'red', 'blue']
encoder = LabelEncoder()
encoded_data = encoder.fit_transform(categorical_data)


#### Mixed Data Types

# Handle both numerical and categorical
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Define transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['age', 'income']),
        ('cat', OneHotEncoder(), ['gender', 'city'])
    ])


### 5. **Real-world Dataset Examples**


# Example 1: Titanic Survival Prediction
from sklearn.datasets import fetch_openml
titanic = fetch_openml('titanic', version=1, as_frame=True)
df = titanic.frame

# Example 2: Credit Card Fraud Detection
# Features: transaction amount, location, time, etc.
# Target: fraudulent (1) or legitimate (0)

# Example 3: Medical Diagnosis
# Features: patient age, symptoms, test results
# Target: disease presence (yes/no)
```

### 6. **Data Preparation Guidelines**

```python
# Check for missing values
print("Missing values:")
print(df.isnull().sum())

# Handle missing values
from sklearn.impute import SimpleImputer

# For numerical features
num_imputer = SimpleImputer(strategy='median')
# For categorical features
cat_imputer = SimpleImputer(strategy='most_frequent')

# Remove outliers (optional)
from scipy import stats
df = df[(np.abs(stats.zscore(df.select_dtypes(include=[np.number]))) < 3).all(axis=1)]
```

## Key Advantages of Random Forest for Different Data Types:

1. **Handles mixed data types** well
2. **Robust to outliers** and missing values
3. **No need for feature scaling**
4. **Works well with high-dimensional data**
5. **Provides feature importance scores**

## When to Use Random Forest:

- ✅ **Classification and regression tasks**
- ✅ **Medium to large datasets**
- ✅ **Datasets with mixed data types**
- ✅ **When interpretability is important**
- ✅ **When you need feature importance**

## When to Avoid Random Forest:

- ❌ **Very high-dimensional sparse data**
- ❌ **When computational efficiency is critical**
- ❌ **When you need probabilistic predictions**
- ❌ **Extremely large datasets** (consider GPU acceleration)

This code provides a solid foundation for implementing Random Forest with various types of data.
 You can adapt it to your specific dataset and problem requirements.