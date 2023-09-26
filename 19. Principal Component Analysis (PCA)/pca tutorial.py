# -*- coding: utf-8 -*-
"""
Created on Sat May 13 02:32:47 2023

@author: yasin
"""

from sklearn.datasets import load_iris
import pandas as pd

# %%

iris = load_iris()

data = iris.data
feature_names = iris.feature_names
y = iris.target

df = pd.DataFrame(data, columns=feature_names)
df['sinif'] = y

x = data

# %% PCA

from sklearn.decomposition import PCA

pca = PCA(n_components=2,whiten=True) # whiten = normalize
pca.fit(x)

x_pca = pca.transform(x)

print('variance ratio: ',pca.explained_variance_ratio_) 
print('sum: ',sum(pca.explained_variance_ratio_)) # verilerin sum:  % 977685206318795'i korunmuş

# %% 2D

df['p1'] = x_pca[:,0]
df['p2'] = x_pca[:,1]

color = ['red','green','blue']

import matplotlib.pyplot as plt

for i in range(3):
    plt.scatter(df.p1[df.sinif == i], df.p2[df.sinif == i], color=color[i], label = iris.target_names[i])
    
plt.legend()
plt.xlabel('p1')
plt.ylabel('p2')
plt.show()

    
    
















