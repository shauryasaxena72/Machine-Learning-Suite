# 📊 Exploratory Data Analysis — 2020–2025 Dataset

## 📌 Overview

This project performs **Exploratory Data Analysis (EDA)** on a dataset containing country-level numerical data from **2020 to 2025**.

The analysis focuses on understanding the dataset structure, handling data-quality issues, identifying outliers and missing values, exploring distributions and relationships, analyzing correlations, and standardizing numerical features.

## 🎯 Objectives

* Understand the structure and characteristics of the dataset
* Inspect dataset dimensions, data types, and statistics
* Identify and remove duplicate records
* Detect potential outliers using the **IQR method**
* Identify and handle missing values
* Perform univariate and bivariate analysis
* Analyze correlations between numerical variables
* Standardize numerical features

## 🛠️ Technologies Used

| Technology       | Purpose                        |
| ---------------- | ------------------------------ |
| **Python**       | Core programming language      |
| **Pandas**       | Data manipulation and analysis |
| **NumPy**        | Numerical operations           |
| **Matplotlib**   | Data visualization             |
| **Seaborn**      | Statistical visualization      |
| **Scikit-learn** | Feature standardization        |

## 🔍 EDA Workflow

### 1. Dataset Loading & Inspection

The dataset is loaded using Pandas and examined using:

* `head()`
* `shape`
* `info()`
* `describe()`

This provides an initial understanding of the dataset's structure, dimensions, data types, and statistical properties.

### 2. Duplicate Detection

Duplicate rows are identified using `duplicated()`.

Detected duplicates are removed using:

```python
df = df.drop_duplicates()
```

### 3. Missing Value Analysis

Missing values are identified across all columns to determine whether the dataset contains incomplete records.

The analysis includes:

* Counting missing values
* Calculating missing-value percentages
* Identifying columns affected by missing data
* Applying appropriate handling techniques

### 4. Outlier Detection

Potential outliers are detected using the **Interquartile Range (IQR)** method.

The IQR is calculated as:

```text
IQR = Q3 − Q1
```

Values outside the following range are considered potential outliers:

```text
Lower Bound = Q1 − 1.5 × IQR
Upper Bound = Q3 + 1.5 × IQR
```

This helps identify unusually high or low observations in numerical variables.

### 5. Univariate Analysis

Individual variables are analyzed to understand their distributions and statistical characteristics.

Visualizations include:

* Histograms
* Distribution plots
* Box plots

These visualizations help identify:

* Distribution shape
* Central tendency
* Spread
* Skewness
* Potential outliers

### 6. Bivariate Analysis

Relationships between pairs of numerical variables are explored using visualizations such as:

* Scatter plots
* Regression plots
* Comparative distributions

This helps identify possible relationships, trends, and patterns between variables.

### 7. Correlation Analysis

Correlation analysis is performed to measure the strength and direction of relationships between numerical variables.

A **correlation matrix** and **heatmap** are used to visualize these relationships.

Correlation values range from:

```text
-1 → Strong negative relationship
 0 → No linear relationship
+1 → Strong positive relationship
```

### 8. Feature Standardization

Numerical features are standardized using **Scikit-learn**.

Standardization transforms numerical variables so that they have approximately:

* Mean = `0`
* Standard deviation = `1`

Example:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[numerical_columns])
```

## 📈 Visualizations

The EDA includes visualizations designed to provide both distribution-level and relationship-level insights.

### Distribution Analysis

* Histograms
* KDE/distribution plots
* Box plots

### Relationship Analysis

* Scatter plots
* Pairwise comparisons

### Correlation Analysis

* Correlation matrix
* Correlation heatmap

These visualizations make it easier to identify trends, unusual observations, variable relationships, and potential data-quality issues.

## 🧹 Data Preprocessing

The preprocessing workflow includes:

1. Loading the dataset
2. Inspecting the dataset
3. Checking for duplicate records
4. Removing duplicates
5. Checking missing values
6. Detecting outliers
7. Exploring numerical distributions
8. Analyzing relationships between variables
9. Computing correlations
10. Standardizing numerical features

## 📂 Project Structure

```text
├── 2020-2025.csv
├── EDA.ipynb
└── README.md
```

## ✅ Key Outcomes

The EDA provides a structured understanding of the **2020–2025 dataset**, including:

* Dataset structure and dimensions
* Data types and descriptive statistics
* Duplicate records
* Missing-value patterns
* Potential outliers
* Numerical feature distributions
* Relationships between variables
* Correlation patterns
* Standardized numerical features

## 🚀 Conclusion

This project demonstrates a complete **Exploratory Data Analysis workflow** using Python. The analysis establishes a clean and structured foundation for further statistical analysis, machine learning, predictive modelling, or other downstream data-science applications.

---

### 👤 Author

**Shaurya Saxena**

**BE (Hons.) CSE — AI & ML**
