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

#%% predicton
import numpy as np

b0 = linear_regression.predict([[0]])
print('b0:',b0)

b0_ = linear_regression.intercept_
print('b0_',b0) #y eksenini kestiği nokta intercept

b1 = linear_regression.coef_
print('b1:',b1) #eğim slope

#maaş = 1663+1138*deneyim
maas_yeni = 1663+1138*11
print(maas_yeni)

print(linear_regression.predict([[11]]))

#%% visualize line
array = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]).reshape(-1,1) #deneyim


y_head = linear_regression.predict(array) #predict edilen maas

plt.scatter(x, y)
plt.plot(array, y_head,color='red')
plt.show()


