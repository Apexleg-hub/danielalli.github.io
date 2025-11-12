import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

# ===============================
# Regression: XGBRegressor
# ===============================

# Load dataset and create DataFrame
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y_reg = data.target

# Split data for regression
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)

# Create and train the regressor with early stopping
regressor = xgb.XGBRegressor(
    objective='reg:squarederror',  # Regression task
    n_estimators=1000,             # Number of trees
    learning_rate=0.1,             # Step size shrinkage
    max_depth=6,                   # Maximum tree depth
    subsample=0.8,                 # Fraction of samples per tree
    colsample_bytree=0.8,          # Fraction of features per tree
    reg_lambda=1,                  # L2 regularization
    early_stopping_rounds=10,      # Early stopping rounds
    random_state=42
)

regressor.fit(
    X_train_reg, y_train_reg,
    eval_set=[(X_test_reg, y_test_reg)],
    verbose=False
)

# Predictions and evaluation for regression
y_pred_reg = regressor.predict(X_test_reg)
rmse = mean_squared_error(y_test_reg, y_pred_reg, squared=False)
r2 = r2_score(y_test_reg, y_pred_reg)
print(f"Regression -> RMSE: {rmse:.4f}, R²: {r2:.4f}")

# ===============================
# Classification: XGBClassifier
# ===============================

# Convert continuous target into binary using the mean as threshold
median_val = y_reg.mean()  # or you can use np.median(y_reg)
y_class = (y_reg > median_val).astype(int)

# Split data for classification
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X, y_class, test_size=0.2, random_state=42)

# Create and train the classifier
classifier = xgb.XGBClassifier(
    objective='binary:logistic',
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    eval_metric='logloss',
    random_state=42
)

classifier.fit(X_train_clf, y_train_clf)
y_pred_clf = classifier.predict(X_test_clf)
print(f"Classification -> Accuracy: {accuracy_score(y_test_clf, y_pred_clf):.4f}")

# ===============================
# Feature Importance Visualization
# ===============================

# Visualize feature importance from the regressor
xgb.plot_importance(regressor)
plt.title("Feature Importance - XGBRegressor")
plt.show()

# ===============================
# Hyperparameter Tuning with RandomizedSearchCV
# ===============================

# Define parameter grid for the regressor (note: remove early_stopping_rounds for tuning)
param_grid = {
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 500, 1000],
    'subsample': [0.6, 0.8, 1.0]
}

# Use a new instance without early stopping for hyperparameter tuning
regressor_tune = xgb.XGBRegressor(
    objective='reg:squarederror',
    random_state=42
)

# Prepare a validation set to be passed to fit via fit_params
fit_params = {
    "eval_set": [(X_test_reg, y_test_reg)],
    "early_stopping_rounds": 10,
    "verbose": False
}

search = RandomizedSearchCV(regressor_tune, param_grid, n_iter=10, scoring='neg_mean_squared_error', cv=3, random_state=42)
search.fit(X_train_reg, y_train_reg, **fit_params)

print("Best parameters found:", search.best_params_)
