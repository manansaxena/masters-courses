'''
Clustering on a given compositonal data
'''


import plotly.express as px
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import  KMeans

data = pd.read_csv("./data/hw2p1_data.csv")
data.head()

fig = px.scatter_ternary(data, a="V1", b="V2", c="V3")
#fig.show()

data = data + 0.1
alr_transform_matrix = np.array([[1,0,-1],[0,1,-1]])
transformed_values = alr_transform_matrix @ np.log(data)

#plt.scatter(transformed_values[0,:], transformed_values[1,:])

kmeans = KMeans(n_clusters = 3, random_state=0)
kmeans.fit(transformed_values.T)
# plt.scatter(transformed_values[0,:], transformed_values[1,:],c=kmeans.labels_)
# plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],c = 'red')

np.savetxt("saxena-hw2p1-predictions.csv", (kmeans.labels_+1))