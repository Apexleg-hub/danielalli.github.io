'''Below is an example using Python’s scikit-learn library to demonstrate Gradient Boosting on a synthetic classification dataset. In this example,
 we generate a dataset, split it into training and testing sets, train a Gradient Boosting classifier, and then evaluate its accuracy.'''


import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate a synthetic binary classification dataset
X, y = make_classification(n_samples=1000,      # total samples
                           n_features=20,       # total features
                           n_informative=15,    # informative features
                           n_redundant=5,       # redundant features
                           random_state=42)

# Split the data into training and testing sets (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Gradient Boosting Classifier
gb_clf = GradientBoostingClassifier(n_estimators=100,  # number of boosting rounds
                                    learning_rate=0.1,   # contribution of each tree
                                    max_depth=3,         # depth of individual trees
                                    random_state=42)

# Train the model
gb_clf.fit(X_train, y_train)

# Predict on the test set
y_pred = gb_clf.predict(X_test)

# Evaluate model accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")


### Explanation
'''- **Data Generation:** We use `make_classification` to create a dataset with 1,000 samples and 20 features, of which 15 are informative.
- **Data Splitting:** The dataset is split into training and testing sets.
- **Model Initialization:** A `GradientBoostingClassifier` is set up with 100 trees, a learning rate of 0.1, and a maximum tree depth of 3.
- **Model Training & Evaluation:** The model is trained on the training set and then used to predict the test set labels. Finally, we compute and print the accuracy of the model.

This example demonstrates how Gradient Boosting builds a strong classifier by sequentially adding weak learners that correct the mistakes of previous models. '''