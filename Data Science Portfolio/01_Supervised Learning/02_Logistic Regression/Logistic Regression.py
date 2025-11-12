'''Here's a simple explanation of **Logistic Regression**, along with code examples and use cases:

---

### **What is Logistic Regression?**  
**Logistic Regression** is a **classification algorithm** (not regression!) used to predict **binary outcomes** 
(yes/no, 0/1, spam/not spam). It estimates the probability that an input belongs to a specific class.  

**How it works**:  
1. It uses the **sigmoid function** to squash predictions into a range between 0 and 1.  
2. If the probability is **> 0.5**, the output is class "1"; otherwise, class "0".'''

#---

### **Code Example**  
#Let’s predict whether a tumor is **malignant** (1) or **benign** (0) using the Breast Cancer dataset:  


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Load dataset
data = load_breast_cancer()
data
X = data.data  # Features (e.g., tumor radius, texture)
y = data.target  # Target (0=benign, 1=malignant)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train with more iterations and scaled data
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Predict
y_pred = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]  # Probabilities of class "1"

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)





#---

### **When to Use Logistic Regression**  
'''- **Binary classification problems**:  
  - Medical diagnosis (e.g., diabetes yes/no).  
  - Spam detection (spam/not spam).  
  - Customer churn prediction (stay/leave).  

- **Multi-class classification** (with extensions like one-vs-rest):  
  - Handwritten digit recognition (0-9).  
  - Iris flower species classification.''' 

#---

### **Key Features**  
'''1. **Probabilistic Output**: Returns probabilities (e.g., 80% chance of being spam).  
2. **Interpretability**: Coefficients show how features affect the outcome.  
3. **Efficiency**: Works well with small-to-medium datasets.''' 

#---

### **Example Datasets**  
'''1. **Bank Customer Churn**: Predict if a customer will leave (0) or stay (1) based on account activity.  
2. **Email Classification**: Classify emails as spam (1) or not spam (0).  
3. **Credit Risk**: Approve (1) or reject (0) a loan application.'''  

#---

'''### **Logistic Regression vs. Linear Regression**  
| **Aspect**          | **Logistic Regression**           | **Linear Regression**          |  
|----------------------|-----------------------------------|---------------------------------|  
| **Output**           | Probability (0-1)                 | Continuous value (e.g., price)  |  
| **Use Case**         | Classification                    | Regression                      |  
| **Equation**         | Sigmoid function                  | Straight line                   |  '''

#---

### **Limitations**  
'''- Assumes a **linear decision boundary** between classes.  
- Struggles with highly complex/non-linear relationships (use SVM or neural networks instead).  

Logistic Regression is a foundational tool for classification tasks—simple, fast, and interpretable! 🎯'''