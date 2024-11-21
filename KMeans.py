import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

iris = load_iris()
X = iris.data
y = iris.target

kmeans = KMeans(n_clusters=3, random_state=42)  # 3 clusters for 3 species
kmeans.fit(X)

centroids = kmeans.cluster_centers_
labels = kmeans.labels_

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap=plt.cm.Paired, marker='o', edgecolors='k', s=100, alpha=0.6)
plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='x', s=200, label='Centroids')
plt.title("K-Means Clustering (Iris Dataset)")
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.legend()
plt.colorbar()
plt.show()

print("Cluster Centers (Centroids):\n", centroids)
print("\nCluster Labels for each data point:\n", labels)

inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()
