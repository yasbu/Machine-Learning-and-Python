# -*- coding: utf-8 -*-
"""
Created on Tue May  2 13:26:31 2023

@author: yasin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% create dataset

#class1
x1 = np.random.normal(25,25,1000) #25 ortalama, 5 sigma, 1000 değer
y1 = np.random.normal(25,5,1000)

#class2
x2 = np.random.normal(55,25,1000) #25 ortalama, 5 sigma, 1000 değer
y2 = np.random.normal(60,5,1000)

#class3
x3 = np.random.normal(55,25,1000) #25 ortalama, 5 sigma, 1000 değer
y3 = np.random.normal(15,5,1000)

x = np.concatenate((x1,x2,x3),axis=0)
y = np.concatenate((y1,y2,y3),axis=0)

dictionary = {'x':x,'y':y}

data = pd.DataFrame(dictionary)

# %%

plt.scatter(x1,y1)
plt.scatter(x2,y2)
plt.scatter(x3,y3)
plt.show()

# %% k-means'ın göreceği şekil

plt.scatter(x1,y1,color='black')
plt.scatter(x2,y2,color='black')
plt.scatter(x3,y3,color='black')
plt.show()

# %% K-MEANS

from sklearn.cluster import KMeans
wcss = []

for k in range(1,15):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,15),wcss)
plt.xlabel('number of k (cluster) value')
plt.ylabel('wcss')
plt.show()

# %% k=5 için model

kmeans2 = KMeans(n_clusters=5)

clusters = kmeans2.fit_predict(data) # fit edip datada uygulamak için fit_predict

data['label'] = clusters

plt.scatter(data.x[data.label==0],data.y[data.label==0])
plt.scatter(data.x[data.label==1],data.y[data.label==1])
plt.scatter(data.x[data.label==2],data.y[data.label==2])
plt.scatter(data.x[data.label==3],data.y[data.label==3])
plt.scatter(data.x[data.label==4],data.y[data.label==4])

plt.scatter(kmeans2.cluster_centers_[:,0],kmeans2.cluster_centers_[:,1])

plt.show()



































