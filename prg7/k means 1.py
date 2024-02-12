import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import pandas as pd

X=pd.read_csv("C:/Users/MCA/Downloads/driver-data.csv")
#no need any preprocessig data is said to be already cleaned
#print(X.info())
x1=X['mean_dist_day'].values
#print(x1)
x2=X['mean_over_speed_perc'].values
#print(x2)

X=np.array(list(zip(x1,x2)))
#print(X.shape)

# plt.plot()
# plt.xlim([0,250])
# plt.ylim([0,100])
# plt.scatter(x1,x2)
# plt.show()


import matplotlib.pyplot as plt1
kmeans=KMeans(n_clusters=3)
kmeans.fit(X)
print(kmeans.cluster_centers_)
print(kmeans.labels_)
plt.title("KMEANS")
plt1.scatter(X[:,0],X[:,1], c=kmeans.labels_,cmap='rainbow')
plt1.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],color='black')
plt1.show()