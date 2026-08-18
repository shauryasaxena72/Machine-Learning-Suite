# Logistic Regression on 2020-2025 Dataset
## This Contains:
### 1. logistic_regression.py
### 2. README File
### 3. Dataset (diabetes.csv)
```python
Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
```
## Load Dataset
```python
df = pd.read_csv("diabetes.csv")
print(df.head())
print("Shape:", df.shape)
print(df.info())
print(df.describe())
```
## Features and Target
```python
X = df.drop("Outcome", axis=1)   # Features
y = df["Outcome"]                # Target variable
```
## Train-Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```
## Logistic Regression Model
```python
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
```
## Predictions
```python
y_pred = model.predict(X_test)
```
## Model Evaluation
```python
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
```
## Output Example
```lua
✅ Accuracy: 0.77
Confusion Matrix:
 [[88 12]
  [21 33]]

Classification Report:
              precision    recall  f1-score   support
           0       0.81      0.88      0.84       100
           1       0.73      0.61      0.66        54
    accuracy                           0.77       154
```
