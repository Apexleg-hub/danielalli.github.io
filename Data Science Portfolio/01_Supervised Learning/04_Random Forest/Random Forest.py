'''Below is an example using the classic Iris dataset, which contains measurements of iris flowers (features) 
and their corresponding species (target). We’ll build a Random Forest classifier, 
analyze its performance, and look at feature importances.'''


### 1. Loading and Preparing the Data

#We’ll use the Iris dataset from scikit-learn, convert it into a DataFrame for easy viewing, and then split it into training and testing sets.


import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load the dataset
iris = load_iris()
X = iris.data       # features
y = iris.target     # target labels

# Optionally, create a DataFrame for visualization
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
print(df.head())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


### 2. Building and Training the Random Forest Model

#We build a RandomForestClassifier, fit it on our training data, and make predictions on the test set.


from sklearn.ensemble import RandomForestClassifier

# Initialize the model with 100 trees and a fixed random state for reproducibility
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)


### 3. Evaluating the Model

#We assess the model’s performance using a classification report and a confusion matrix.


from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Print classification metrics
print(classification_report(y_test, y_pred))

# Generate and plot the confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()




### 4. Analyzing Feature Importance

#Random Forests provide a measure of the importance of each feature. Here’s how you can visualize which features are most influential in predicting the iris species.


# Calculate feature importances
importances = clf.feature_importances_
feature_names = iris.feature_names

# Create a Series and plot
feature_importances = pd.Series(importances, index=feature_names)
feature_importances.sort_values().plot(kind='barh')
plt.xlabel('Importance Score')
plt.title('Feature Importances')
plt.show()




### Summary

'''1. **Data Preparation:**  
   We loaded the Iris dataset and split it into training and testing sets.

2. **Model Building:**  
   A Random Forest classifier was built with 100 trees.

3. **Model Evaluation:**  
   The classifier’s performance was evaluated using a classification report and confusion matrix.

4. **Feature Analysis:**  
   We visualized the importance of each feature to understand which measurements were most influential in classification.

This example demonstrates how to apply Random Forest to structured data using Python, providing both predictive modeling and interpretability through feature importance.'''