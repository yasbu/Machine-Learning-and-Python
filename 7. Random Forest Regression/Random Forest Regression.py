# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 18:32:00 2023

@author: yasin
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('random+forest+regression+dataset.csv', sep=';', header=None)

x = df.iloc[:,0].values.reshape(-1,1)
y = df.iloc[:,1].values.reshape(-1,1)

#%% 
from sklearn.ensemble import RandomForestRegressor
rf= RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(x,y)

print('7.8 seviyesindeki fiyat:',rf.predict([[7.8]]))

x_ = np.arange(min(x),max(x),0.01).reshape(-1,1)
y_head = rf.predict(x_)

#%% visualize
plt.scatter(x, y,color='red')
plt.plot(x_,y_head, color='green')
plt.xlabel('tribun level')
plt.ylabel('ucret')
plt.show()
