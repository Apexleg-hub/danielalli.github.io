#### 3. **Classification Example**
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb

# Load classification data
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create XGBoost classifier
clf = xgb.XGBClassifier(
    objective='binary:logistic',  # Binary classification
    n_estimators=200, # Number of trees
    learning_rate=0.05, # Step size shrinkage
    max_depth=3,
    eval_metric='logloss'         # Evaluation metric
    )

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")