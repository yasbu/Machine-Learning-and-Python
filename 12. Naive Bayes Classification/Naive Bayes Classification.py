# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 06:41:44 2023

@author: yasin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%
data = pd.read_csv('data.csv')

#%%
data.drop(['id','Unnamed: 32'],axis=1,inplace=True)
data.tail()
# malignant = M kötü huylu tumor
# benign = B iyi huylu tumor

M = data[data.diagnosis=='M']
B = data[data.diagnosis=='B']

#%% 
# scatter plot
plt.scatter(M.radius_mean,M.area_mean,color='red',label='bad')
plt.scatter(B.radius_mean,B.area_mean,color='green',label='good')
plt.legend()
plt.show()

# scatter plot
plt.scatter(M.radius_mean,M.texture_mean,color='red',label='bad')
plt.scatter(B.radius_mean,B.texture_mean,color='green',label='good')
plt.xlabel('radius_mean')
plt.ylabel('texture_mean')
plt.legend()
plt.show()

#%%
data.diagnosis = [1 if i == 'M' else 0 for i in data.diagnosis]

y = data.diagnosis.values
x_data = data.drop(['diagnosis'],axis=1)

# normalization

x = (x_data - np.min(x_data))/(np.max(x_data) - np.min(x_data))

#%%
#train test split
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=1)

#%% Naive Bayes

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
nb.fit(x_train,y_train)

#%%
print('print accuracy of svm algo: ',nb.score(x_test,y_test))



























