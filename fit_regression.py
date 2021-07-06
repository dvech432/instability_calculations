# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 15:45:08 2021

@author: vechd
"""

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from matplotlib import pyplot

def fit_regression(Xtrain,Ytrain, Xtest, Ytest):
 reg_model = DecisionTreeRegressor()
 reg_model.fit(Xtrain, Ytrain)
 Y_pred=reg_model.predict(Xtest)

  # get importance
 importance = reg_model.feature_importances_
  # summarize feature importance
 for i,v in enumerate(importance):
   print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
 pyplot.bar([x for x in range(len(importance))], importance)
 pyplot.show()