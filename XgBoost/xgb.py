import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
df=pd.read_csv("diabetes2.csv")
x=df.drop("Outcome",axis=1)
y=df["Outcome"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=XGBClassifier()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print("Accuracy Score : ",accuracy_score(y_test,y_pred))
print("Confusion Matrix : ",confusion_matrix(y_test,y_pred))
print("Classification Report : ",classification_report(y_test,y_pred))
from xgboost import plot_importance

# 1️⃣ Using XGBoost built-in plot
plt.figure(figsize=(10,6))
plot_importance(model, importance_type='weight')  # You can also use 'gain' or 'cover'
plt.title("Feature Importance (XGBoost)")
plt.show()

# 2️⃣ Using pandas for a horizontal bar plot
import pandas as pd
importance = model.feature_importances_
features = x.columns
feature_df = pd.df({'Feature': features, 'Importance': importance})
feature_df = feature_df.sort_values(by='Importance', ascending=True)

plt.figure(figsize=(10,8))
plt.barh(feature_df['Feature'], feature_df['Importance'], color='red')
plt.xlabel("Importance")
plt.title("Feature Importance (Bar Plot)")
plt.show()
