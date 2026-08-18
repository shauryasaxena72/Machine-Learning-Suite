# Simple Linear Regression (SLR) on 2020–2025 Dataset  

## Import Libraries 
```python
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression  
from sklearn.metrics import mean_squared_error, r2_score  
```
## Load Dataset 
```python
df = pd.read_csv("2020-2025.csv")  
```
## Handle Missing Values 
```python
df['2020'].fillna(df['2020'].median(), inplace=True)  
df['2025'].fillna(df['2025'].median(), inplace=True)  
```
## Feature & Target 
```python
X = df['2020'].values  
Y = df['2025'].values  
```
## Reshape Feature
```python
X = X.reshape(-1, 1)  

print("Shape of X:", X.shape)  
print("Shape of Y:", Y.shape)  
```
## Scatter Plot 
```python
plt.scatter(X, Y, color='blue')  
plt.xlabel("2020 Values")  
plt.ylabel("2025 Values")  
plt.title("Scatter Plot of 2020 vs 2025 Values")  
plt.show()  
```
## Train-Test Split 
```python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)  
```
## Train Linear Regression Model 
```python
model = LinearRegression()  
model.fit(X_train, Y_train)  
```
## Model Parameters 
```python
print("Slope (Coefficient):", model.coef_[0])  
print("Intercept:", model.intercept_)  
```
## Predictions 
```python
Y_pred = model.predict(X_test)  
```
## Model Evaluation 
```python
rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))  
r2 = r2_score(Y_test, Y_pred)  
print("RMSE:", rmse)  
print("R² Score:", r2)  
```
## Regression Line Plot
```python
plt.scatter(X, Y, color='blue', label="Actual")  
plt.plot(X, model.predict(X), color='red', label="Regression Line")  
plt.xlabel("2020 Values")  
plt.ylabel("2025 Values")  
plt.title("2020 vs 2025 with Regression Line")  
plt.legend()  
plt.show()
```

# SLR Completed
```python
print("Simple Linear Regression Completed Successfully!")  
```
