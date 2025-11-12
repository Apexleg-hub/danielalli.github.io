### **Hyperparameter Tuning**
#Use `GridSearchCV` or `RandomizedSearchCV` for optimization:

from sklearn.model_selection import RandomizedSearchCV

param_grid = {
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 500, 1000],
    'subsample': [0.6, 0.8, 1.0]}

search = RandomizedSearchCV(model, param_grid, n_iter=10, scoring='neg_mean_squared_error', cv=3)
search.fit(X_train, y_train)

print("Best parameters:", search.best_params_)