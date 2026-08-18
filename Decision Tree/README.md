# Diabetes Prediction using Decision Tree Classifier
## Project Overview
### This project implements a Decision Tree Classifier to predict whether a patient has diabetes based on diagnostic measurements. It uses a publicly available dataset containing features such as glucose level, blood pressure, BMI, and more.

The main goal is to analyze the dataset, train a Decision Tree model, and evaluate its performance using accuracy, confusion matrix, and classification report.
## Dataset
### Filename: diabetes2.csv
### Description: The dataset contains health metrics of patients along with a binary target column Outcome:
### 0 – No Diabetes
### 1 – Diabetes
## Features:
### 
Pregnancies
Glucose
BloodPressure
SkinThickness
Insulin
BMI
DiabetesPedigreeFunction
Age
Requirements

## Code Overview
### Importing Files
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
```
### Load the Dataset
```python
df = pd.read_csv("diabetes2.csv")
```
### Define Features and Target
```python
X = df.drop("Outcome", axis=1)
y = df["Outcome"]
```
### Train-Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```
### Train Decision Tree Classifier
```python
model = DecisionTreeClassifier(criterion="gini", max_depth=4, random_state=42)
model.fit(X_train, y_train)
```
### Make Predictions
```python
y_pred = model.predict(X_test)
```
### Evaluate Model
```python
print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```
### Visualize Decision Tree
```python
plt.figure(figsize=(15,8))
plot_tree(model, feature_names=X.columns, class_names=["No Diabetes", "Diabetes"], filled=True)
plt.show()
```
## Output
```lau
Accuracy: 0.6948051948051948

Confusion Matrix:
 [[69 30]
 [17 38]]

Classification Report:
               precision    recall  f1-score   support

           0       0.80      0.70      0.75        99
           1       0.56      0.69      0.62        55

    accuracy                           0.69       154
   macro avg       0.68      0.69      0.68       154
weighted avg       0.72      0.69      0.70       154
```
