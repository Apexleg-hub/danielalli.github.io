#Here’s a **complete and beginner-friendly Python example** of using **K-Nearest Neighbors (KNN)** 
# for classification with the **Iris dataset** 

---

###  Example: K-Nearest Neighbors (KNN) Classification


# Import required libraries
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load dataset (Iris dataset)
iris = datasets.load_iris()
X = iris.data     # features
y = iris.target   # labels

# 2. Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 3. Standardize features (very important for KNN)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Create and train the KNN model
knn = KNeighborsClassifier(n_neighbors=5)  # You can change the value of k
knn.fit(X_train, y_train)

# 5. Make predictions
y_pred = knn.predict(X_test)

# 6. Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))


###  Explanation

#1. **Load dataset** – uses the built-in Iris dataset from `sklearn.datasets`.
#2. **Split the data** – separates into training (70%) and testing (30%).
3. **Standardize features** – scales data so that distance-based algorithms like KNN work properly.
4. **Train the model** – fits a KNN classifier with `k=5`.
5. **Predict & evaluate** – checks model performance.



### Tip: Try tuning `k`

#You can find the best `k` (number of neighbors) with a loop:


import matplotlib.pyplot as plt

accuracy_scores = []

for k in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy_scores.append(accuracy_score(y_test, y_pred))

plt.plot(range(1, 21), accuracy_scores, marker='o')
plt.title('KNN Accuracy for different k values')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
plt.show()



Would you like me to show how to **visualize the decision boundaries** for 
the KNN model (with 2 features)? It makes the concept very easy to understand.
