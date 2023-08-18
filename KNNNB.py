import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load the dataset
data = pd.read_csv("diabetes.csv")

# Step 2: Implement Standard Scaler
def custom_standard_scaler(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    scaled_X = (X - mean) / std
    return scaled_X

# Step 3: Scale Data
features = data.iloc[:, :-1].values
scaled_features = custom_standard_scaler(features)

# Step 4: Split into Training and Testing Sets
labels = data.iloc[:, -1].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(scaled_features, labels, test_size=0.2, random_state=42)

# Step 5: Determine the Best K Value
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

k_values = list(range(1, 21))
accuracy_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)

best_k = k_values[np.argmax(accuracy_scores)]

plt.plot(k_values, accuracy_scores)
plt.xlabel('K Value')
plt.ylabel('Accuracy')
plt.title('KNN Accuracy for Different K Values')
plt.show()

# Step 6: Run 5-fold Cross-Validation
from sklearn.model_selection import cross_val_score

knn_cv = KNeighborsClassifier(n_neighbors=best_k)
cv_scores = cross_val_score(knn_cv, scaled_features, labels, cv=5)
mean_cv_accuracy = np.mean(cv_scores)
std_cv_accuracy = np.std(cv_scores)

# Step 7: Evaluate Using Confusion Matrix
knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X_train, y_train)
y_pred_best = knn_best.predict(X_test)

from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_test, y_pred_best)

# Step 8: Explain Model Accuracy
accuracy_explanation = "The model achieved an accuracy of {:.2f}% on the test set. The confusion matrix:\n{}".format(
    accuracy_score(y_test, y_pred_best) * 100, conf_matrix)

# Step 9: Retrain Using Leave-One-Out Cross-Validation
from sklearn.model_selection import LeaveOneOut

loo = LeaveOneOut()
loo_scores = cross_val_score(knn_cv, scaled_features, labels, cv=loo)
mean_loo_accuracy = np.mean(loo_scores)
std_loo_accuracy = np.std(loo_scores)

print("Best K value:", best_k)
print("5-fold Cross-Validation Mean Accuracy:", mean_cv_accuracy)
print("5-fold Cross-Validation StdDev Accuracy:", std_cv_accuracy)
print("Leave-One-Out Cross-Validation Mean Accuracy:", mean_loo_accuracy)
print("Leave-One-Out Cross-Validation StdDev Accuracy:", std_loo_accuracy)
print(accuracy_explanation)
