# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#import library
import pandas as pd
import matplotlib.pyplot as plt

#import data
df= pd.read_csv('linear_regression_dataset.csv', sep=';')

#plot data
plt.scatter(df.deneyim, df.maas)
plt.xlabel('deneyim')
plt.ylabel('maas')
plt.show()

#%% Linear Regression

#sklearn library
from sklearn.linear_model import LinearRegression

#linear regression model
linear_regression = LinearRegression()

x = df.deneyim.values.reshape(-1,1)
y = df.maas.values.reshape(-1,1)

linear_regression.fit(x, y)

y_head = linear_regression.predict(x) #predict edilen maas


#%%
from sklearn.metrics import r2_score

print('r_square score:', r2_score(y,y_head))


