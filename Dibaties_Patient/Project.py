import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

data=pd.read_csv('/home/sumit_singh_rajput/Documents/Dibaties_Patient/DB_Patient/diabetes.csv')
numerical_features = data.select_dtypes(include=['int64', 'float64']).columns
categorical_features = data.select_dtypes(exclude=['int64', 'float64']).columns
X = data[numerical_features]
y = data['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Pre-processing with StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build and train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict on the training and testing data
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculate classification metrics
def calculate_metrics(y_true, y_pred, label):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=label)
    recall = recall_score(y_true, y_pred, pos_label=label)
    f1 = f1_score(y_true, y_pred, pos_label=label)
    confusion = confusion_matrix(y_true, y_pred)
    return accuracy, precision, recall, f1, confusion

# Calculate metrics for the training and testing data
train_accuracy, train_precision, train_recall, train_f1, train_confusion = calculate_metrics(y_train, y_train_pred, label=1)
test_accuracy, test_precision, test_recall, test_f1, test_confusion = calculate_metrics(y_test, y_test_pred, label=1)

# Print the classification metrics
print("Training Set Metrics:")
print(f"Accuracy: {train_accuracy:.2f}")
print(f"Precision: {train_precision:.2f}")
print(f"Recall: {train_recall:.2f}")
print(f"F1 Score: {train_f1:.2f}")

print("\nTesting Set Metrics:")
print(f"Accuracy: {test_accuracy:.2f}")
print(f"Precision: {test_precision:.2f}")
print(f"Recall: {test_recall:.2f}")
print(f"F1 Score: {test_f1:.2f}")