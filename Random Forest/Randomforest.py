import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
df=pd.read_csv("diabetes2.csv")
x=df.drop("Outcome",axis=1)
y=df["Outcome"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print("Accuracy Score : ",accuracy_score(y_test,y_pred))
print("Confusion Matrix : ",confusion_matrix(y_test,y_pred))
print("Classification Report : ",classification_report(y_test,y_pred))
importances = model.feature_importances_
features = x.columns

plt.figure(figsize=(8,5))
sns.barplot(x=importances, y=features, palette="viridis")
plt.title("Random Forest Feature Importance")
plt.show()
