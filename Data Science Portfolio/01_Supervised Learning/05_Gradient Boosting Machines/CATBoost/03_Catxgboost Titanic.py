# CatBoost, short for Categorical Boosting, is an open-source gradient boosting library developed by Yandex. 
# It is designed to handle both categorical and numerical features efficiently, making it a powerful tool for classification and regression tasks. 
# One of the standout features of CatBoost is its ability to handle categorical features without requiring preprocessing steps 
# like one-hot encoding or label encoding.

#---
# Key Features of CatBoost

#Handling Categorical Features: CatBoost automatically handles categorical features, eliminating the need for manual encoding. 
# This simplifies the data preparation process and reduces the risk of overfitting.

# Built-in Methods for Missing Values: CatBoost can handle missing values in the input data without requiring imputation, thanks
#  to its symmetric weighted quantile sketch (SWQS) algorithm.

# Robust to Overfitting: CatBoost employs techniques like ordered boosting and random permutations for feature combinations to prevent overfitting.

# Fast and Scalable: CatBoost offers a GPU-accelerated version, allowing for quick training on large datasets.

# Example: Using CatBoost for Classification

# Here is a step-by-step guide to using CatBoost for a classification task using the Titanic dataset:
#----

# Step 1: Install CatBoost

# pip install catboost
#Step 2: Import Necessary Libraries

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

#Step 3: Load and Preprocess the Dataset

# Load the dataset
titanic_data = pd.read_csv('titanic.csv')
titanic_data = titanic_data.drop(['Name', 'Ticket', 'Cabin'], axis=1)

# Handle missing values
titanic_data['Age'].fillna(titanic_data['Age'].mean(), inplace=True)
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace=True)

# Encode categorical variables
le = LabelEncoder()
titanic_data[['Sex', 'Embarked']] = titanic_data[['Sex', 'Embarked']].apply(le.fit_transform)

# Split the data into features and target
X = titanic_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
y = titanic_data['Survived']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Step 4: Train the CatBoost Model

# Initialize the CatBoostClassifier
model = CatBoostClassifier(iterations=100, learning_rate=0.1, depth=6)

# Fit the model to the training data
model.fit(X_train, y_train)

#Step 5: Evaluate the Model

# Predict on the testing data
y_pred = model.predict(X_test)

# Calculate accuracy and classification report
accuracy = accuracy_score(y_test, y_pred)
classification_report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report)

#Step 6: Feature Importance

# Get feature importance
feature_importance = model.get_feature_importance()
feature_names = X.columns

# Plot feature importance
import matplotlib.pyplot as plt
plt.bar(feature_names, feature_importance)
plt.xlabel("Feature Importance")
plt.title("CatBoost Feature Importance")
plt.show()

#Conclusion

# CatBoost is a versatile and powerful gradient boosting library that simplifies handling categorical features and missing values. 
# It provides robust performance with minimal parameter tuning and supports GPU acceleration for faster training.
# Despite its advantages, users should be aware of its memory consumption and training time, especially for large datasets.