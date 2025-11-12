
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load your data (replace 'your_data.csv' with your file path)
data = pd.read_csv('your_data.csv')

# Separate features (X) and target variable (y)
X = data.drop('target_variable', axis=1)  # Replace 'target_variable' with your target column name
y = data['target_variable']

# Identify categorical features (replace with your actual categorical column names)
categorical_features = [col for col in X.columns if X[col].dtype == 'object'] #identifies object columns

#Convert categorical columns to category type.
for col in categorical_features:
    X[col] = X[col].astype('category')

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the CatBoost classifier
model = CatBoostClassifier(
    iterations=100,  # Number of boosting iterations
    learning_rate=0.1, # Learning rate
    depth=6, # Depth of trees
    loss_function='Logloss',  # Or 'MultiClass' for multiclass classification
    verbose=10, # Print metrics every 10 iterations
    random_state=42, #For reproducibility.
    cat_features=categorical_features #specify the categorical columns.
)

model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

#Example of predicting on new data.
new_data = pd.DataFrame({
    'feature1': [value1],
    'feature2': [value2],
    # ... other features
})

#Ensure the same categorical types exist in the new data.
for col in categorical_features:
    new_data[col] = new_data[col].astype('category')

new_predictions = model.predict(new_data)
print(f'New Predictions: {new_predictions}')



#**Explanation and Key Points:**

'''1.  **Import Libraries:** Imports necessary libraries (pandas, CatBoost, scikit-learn).
2.  **Load Data:** Loads your data from a CSV file (replace `'your_data.csv'` with your file).
3.  **Separate Features and Target:** Splits the data into features (X) and the target variable (y).
4.  **Identify Categorical Features:** Identifies categorical columns. It is crucial that the cat_features parameter in the 
    CatBoostClassifier is populated with the correct column names.
5.  **Convert to category type:** Catboost works best when categorical features are of the category type.
6.  **Split Data:** Splits the data into training and testing sets.
7.  **Initialize CatBoost Classifier:**
    * Creates an instance of `CatBoostClassifier`.
    * `iterations`: Sets the number of boosting iterations.
    * `learning_rate`: Controls how much the model learns in each iteration.
    * `depth`: Sets the depth of the trees.
    * `loss_function`: Specifies the loss function (e.g., 'Logloss' for binary classification, 'MultiClass' for multiclass).
    * `verbose`: Controls the verbosity of the output.
    * `random_state`: Ensures reproducibility.
    * `cat_features`: Specifies the list of categorical feature names. This is the most important part for handling categorical data directly.
8.  **Train the Model:** Trains the CatBoost model using the training data.
9.  **Make Predictions:** Predicts the target variable for the test set.
10. **Evaluate Accuracy:** Calculates and prints the accuracy of the model.
11. **Predicting new data:** Demonstrates how to predict on new data. It is very important that new data that contains categorical columns is
     also converted to the category type.'''

#**Important Notes:**

'''* **Replace Placeholders:** Remember to replace `'your_data.csv'`, `'target_variable'`, and the categorical column names with your actual data and column names.
* **Hyperparameters:** Experiment with different hyperparameters (e.g., `iterations`, `learning_rate`, `depth`) to optimize your model's performance.
* **Data Preprocessing:** While CatBoost handles categorical data well, you might still need to perform other preprocessing steps (e.g., handling missing values, feature scaling) depending on your data.
* **Regression:** For regression tasks, use `CatBoostRegressor` instead of `CatBoostClassifier`.
* **Installation:** Make sure you have the `catboost` library installed (`pip install catboost`).'''
