# Here’s a **simple and clear Python example** of how to use **Support Vector Machine (SVM)** 
# for classification using **scikit-learn (sklearn)** 👇



### 🧠 Example: SVM Classification with Scikit-Learn


# Import necessary libraries
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load dataset (Iris dataset)
iris = datasets.load_iris()
X = iris.data     # features
y = iris.target   # labels

# 2. Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 3. Scale the features (important for SVM performance)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Create and train the SVM model
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')  # You can try 'linear', 'poly', 'sigmoid'
svm_model.fit(X_train, y_train)

# 5. Make predictions
y_pred = svm_model.predict(X_test)

# 6. Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))


---

### ⚙️ How it works

1. **Load dataset:** Uses the Iris dataset from sklearn.
2. **Split data:** 70% for training, 30% for testing.
3. **Scale features:** SVMs work best when features are standardized.
4. **Train model:** With an RBF kernel (default for non-linear data).
5. **Predict & evaluate:** Outputs accuracy, classification report, and confusion matrix.

---

### 🔁 You can change kernels:

```python
SVC(kernel='linear')
SVC(kernel='poly', degree=3)
SVC(kernel='sigmoid')
```

---

Would you like me to show how to **plot the decision boundaries** for visualization?
 It’s very useful if you want to *see* how SVM separates classes.
