import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

d=pd.read_csv("cropyield.csv")
x=d.values

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_scaled=sc.fit_transform(x)

from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    km=KMeans(n_clusters=i,init="k-means++",random_state=0)
    km.fit(x_scaled)
    wcss.append(km.inertia_)
    
plt.plot(range(1,11),wcss)
plt.title("elbow method")
plt.xlabel("the no. of clusters")
plt.ylabel("wcss")
plt.show()

km=KMeans(n_clusters=5,init="k-means++",random_state=0)
y_kmeans=km.fit_predict(x)
print(y_kmeans)

y_pseudo_labels=y_kmeans

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()
scoresclf = cross_val_score(clf, x_scaled, y_pseudo_labels, cv=5)  # 5-fold cross-validation
print("Cross-validation scores for random forest classifier:", scoresclf)
print("Mean Accuracy for random forest classifier:", np.mean(scoresclf) * 100, "%")

from xgboost import XGBClassifier

xgb = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=10)
scores_xgb = cross_val_score(xgb, x_scaled, y_pseudo_labels, cv=5)

print("XGBoost Cross-validation scores:", scores_xgb)
print("Mean Accuracy for XGBoost:", np.mean(scores_xgb) * 100, "%")