# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 10:54:41 2023

@author: yasin
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df = pd.read_csv('polynomial+regression.csv', sep=(';'))

y = df.araba_max_hiz.values.reshape(-1,1)
x = df.araba_fiyat.values.reshape(-1,1)

plt.scatter(x, y)
plt.ylabel('araba_max_hiz')
plt.xlabel('araba_fiyat')
plt.show()

# linear regression = y = b0 + b1*x
# multiple linear regression = y = b0 + b1*x1 + b2*x2

#%% linear regression

lr = LinearRegression()

lr.fit(x,y)

#%%predict
y_head = lr.predict(x)

plt.scatter(x, y)
plt.plot(x,y_head, color='red', label='linear')
plt.show()


print('10 milyonluk arabanın tahmini hızı: ',lr.predict([[10000]]))

#%%
# polynomial linear regression = y = b0 + b1*x + b2*x^2 + b3*x^3+...bn*x^n

from sklearn.preprocessing import PolynomialFeatures

polynomial_regression = PolynomialFeatures(degree=2) #x^2
 
x_polynomial = polynomial_regression.fit_transform(x)

lr2 = LinearRegression()
lr2.fit(x_polynomial,y)

#%%
y_head2 = lr2.predict(x_polynomial)

plt.scatter(x,y)
plt.plot(x,y_head2,color='green',label='poly')
plt.legend()
plt.show()


