# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("2020-2025.csv")
pd.set_option('display.float_format', '{:,.0f}'.format)

# Head, Shape, info, describe
print(df.head())
print("Shape:", df.shape)
print(df.info())
print(df.describe())

# Duplicates
dups = df.duplicated()
print("Duplicate rows:", dups.sum())
print(df[dups])

# Remove duplicates
df = df.drop_duplicates()

# Checking for outliers
sns.boxplot(x=df["2025"])
plt.show()

# Outliers
for col in df.columns[1:]:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    print(f"{col}: {len(outliers)} outliers")

# Check missing values
print(df.isnull().sum())

# Replace missing values with median of each column
for col in df.columns[1:]:  # skip 'Country' column
    df[col] = df[col].fillna(df[col].median())

# Verify missing values are gone
print(df.isnull().sum())

# 5. Univariate Analysis
sns.histplot(df['2025'],bins=20)
plt.title("Distribution of 2025")
plt.show()

# Bivariate Analysis 
sns.scatterplot(x="2020", y="2025", data=df)
plt.title("2020 vs 2025")
plt.show()

# Correlation
print(df.corr(numeric_only=True))
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt='.2f', cmap="coolwarm")
plt.show()

# Standardize and Normalization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
for col in df.columns[1:]:  # skip 'Country' cause it is not a numerical value.
    df[col] = scaler.fit_transform(df[[col]])

print(df.head())
# Import libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("2020-2025.csv")

df['2020'].fillna(df['2020'].median(), inplace=True)
df['2025'].fillna(df['2025'].median(), inplace=True)

# Feature and Target

X = df['2020'].values  # Feature: 2020 values
Y = df['2025'].values  # Target: 2025 values

# Reshape X to 2D
X = X.reshape(-1, 1)

# Check shapes
print("Shape of X:", X.shape)
print("Shape of Y:", Y.shape)

# Scatter Plot

plt.scatter(X, Y, color='blue')
plt.xlabel("2020 Values")
plt.ylabel("2025 Values")
plt.title("Scatter Plot of 2020 vs 2025 Values")
plt.show()

# Train-Test Split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Simple Linear Regression Model

model = LinearRegression()
model.fit(X_train, Y_train)

# Slope and Intercept
print("Slope (Coefficient):", model.coef_[0])
print("Intercept:", model.intercept_)

# Predictions

Y_pred = model.predict(X_test)

# Evaluate the Model

rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
r2 = r2_score(Y_test, Y_pred)
print("RMSE:", rmse)
print("R² Score:", r2)

# Plot Regression Line
plt.scatter(X, Y, color='blue', label='Actual')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.xlabel("2020 Values")
plt.ylabel("2025 Values")
plt.title("2020 vs 2025 Values with Regression Line")
plt.legend()
plt.show()