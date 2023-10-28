import sklearn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as color
import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN, KMeans, SpectralClustering


data = pd.read_csv("./data_hw6_problem1-1.csv")
plt.scatter(data["x1"], data["x2"])

dbscan = DBSCAN(eps=7,min_samples=8)
dbscan.fit(data[["x1","x2"]])

kmeans = KMeans(n_clusters = 5, random_state=0)
kmeans.fit(data[["x1","x2"]])

SpCluster = SpectralClustering(n_clusters=5)
SpCluster.fit(data[["x1","x2"]])

data['DBSCAN_labels']=dbscan.labels_ 
data['kmeans_labels']=kmeans.labels_
data['SpCluster_labels']=SpCluster.labels_

plt.scatter(data["x1"],data["x2"],c=data['DBSCAN_labels'])
plt.scatter(dbscan.cluster_centers_[:,0], dbscan.cluster_centers_[:,1],c = 'red')
plt.scatter(data["x1"], data["x2"],c=data["kmeans_labels"])
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],c = 'red')
plt.scatter(data["x1"], data["x2"],c=data["SpCluster_labels"])
plt.scatter(SpCluster.cluster_centers_[:,0], SpCluster.cluster_centers_[:,1],c = 'red')

