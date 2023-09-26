# -*- coding: utf-8 -*-
"""
Created on Sat May 13 03:25:37 2023

@author: yasin
"""

from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
# %%
iris = load_iris()

x = iris.data
y = iris.target

# %% normalization
x = (x-np.min(x)/np.max(x)-np.min(x))

# %% train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3)

# %% knn model
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3) # k = n_neighbors

# %% K fol CV K=10
from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator= knn, X = x_train, y = y_train, cv=10)
print('accuracy mean: ', np.mean(accuracies))
print('accuracy std: ', np.std(accuracies))

# %%
knn.fit(x_train,y_train)
print('test accuracy: ', knn.score(x_test,y_test))

# %% grid search cross validation (n_neighbors'un hangisinin en iyi sonucu verecek?)
 
from sklearn.model_selection import GridSearchCV

grid = {'n_neighbors':np.arange(1,50)}
knn = KNeighborsClassifier()

knn_cv = GridSearchCV(knn, grid, cv=10) 
knn_cv.fit(x,y)

# %% print hyperparameter KNN algoritmasındaki K değeri
print('tuned hyperparameter K: ', knn_cv.best_params_)
print('tuned parametereye göre en iyi accuracy (best score): ', knn_cv.best_score_)

# %% Grid search CV with logistic regression
# iki adet target gerektiğinden ilk 100 veriyi alacağız

x = x[:100,:]
y = y[:100]

from sklearn.linear_model import LogisticRegression

grid = {'C':np.logspace(-3,3,7),'penalty':['l1','l2']} #c büyükse overfit küçükse underfit olur. l1=lasso,l2=rigde

logreg = LogisticRegression(solver="liblinear")
logreg_cv = GridSearchCV(logreg,grid,cv=10)
logreg_cv.fit(x,y)

print('tuned  hyperparameters: (best parameters): ', logreg_cv.best_params_)
print('accuracy: ',logreg_cv.best_score_)

























