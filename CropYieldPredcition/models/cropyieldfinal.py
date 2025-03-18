import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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

pca = PCA(n_components=2)
x_pca = pca.fit_transform(x_scaled)

plt.figure(figsize=(8, 5))
plt.scatter(x_pca[:, 0], x_pca[:, 1], c=y_kmeans, cmap="viridis", alpha=0.7)
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], s=200, c='red', marker="X", label="Centroids")
plt.title("K-Means Clustering (2D Projection using PCA)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid(True)
plt.show()

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()
scoresclf = cross_val_score(clf, x_scaled, y_pseudo_labels, cv=5)  # 5-fold cross-validation
print("Cross-validation scores for random forest classifier:", scoresclf)
print("Mean Accuracy for random forest classifier:", np.mean(scoresclf) * 100, "%")

plt.figure(figsize=(8, 5))
plt.bar(range(1, 6), scoresclf * 100, color='skyblue', edgecolor='black')
plt.axhline(np.mean(scoresclf) * 100, color='red', linestyle="--", label=f"Mean Accuracy: {np.mean(scoresclf) * 100:.2f}%")
plt.xticks(range(1, 6), labels=[f"Fold {i}" for i in range(1, 6)])
plt.xlabel("Cross-Validation Fold")
plt.ylabel("Accuracy (%)")
plt.title("Cross-Validation Accuracy for Random Forest Classifier")
plt.legend()
plt.grid(axis="y")
plt.show()

clf.fit(x_scaled, y_pseudo_labels) 
feature_importances = clf.feature_importances_
features = [f"Feature {i}" for i in range(len(feature_importances))]

plt.figure(figsize=(10, 5))
plt.barh(features, feature_importances, color="lightcoral", edgecolor="black")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.title("Feature Importance from Random Forest Classifier")
plt.gca().invert_yaxis()
plt.grid(axis="x")
plt.show()