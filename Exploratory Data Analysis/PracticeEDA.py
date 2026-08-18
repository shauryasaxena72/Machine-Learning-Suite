# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv("2020-2025.csv")
print(df.head())
print(df.shape)
print(df.describe())
print(df.info())
dups=df.duplicated()
print("Duplicates : ",dups.sum())
df[dups]
sns.boxplot(data=df)
plt.show()
