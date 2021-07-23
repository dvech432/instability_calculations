
import numpy as np
from scipy.io import loadmat

from gplearn.genetic import SymbolicRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.random import check_random_state
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import graphviz
import pandas as pd
import math 


x = loadmat('D:\Research\Data\DISP\cluster_total.mat')
cc=x['clustertotal']
cc=np.array(cc)

at=np.where(cc[:,22]>-1) # picking cases with significant instability
X=np.concatenate( (cc[at,2], cc[at,10], cc[at,14], cc[at,15]),axis=0).T # generating

x_grid=np.arange(0,len(cc))

at=np.where(cc[:,22]>1e-4) # picking cases with instability
Y_lab=np.array(pd.get_dummies(cc[:,22]>1e-4)[0]) # generating labels

X_inst=np.concatenate( (cc[at,2], cc[at,10], cc[at,14], cc[at,15]),axis=0).T # generating


Y_theta=[]
for i in range(0,len(at[0])):
    Y_theta.append(math.atan2(cc[at[0][i],23],cc[at[0][i],24])*180/math.pi) # k-vector angle

Y_theta=np.array(Y_theta)
Y_mag = np.sqrt(cc[at,23]**2 + cc[at,24]**2).T # magnitude of the k-vector
Y_growth = cc[at,22].T # value of gamma


## split into train and test --- classification vs. regression
from sklearn.model_selection import train_test_split
train, test = train_test_split(x_grid, test_size=0.2)

x_grid_inst = np.arange(0,len(Y_mag))
train_inst, test_inst = train_test_split(x_grid_inst, test_size=0.2)

############## Classification
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from matplotlib import pyplot
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import tree

model = DecisionTreeClassifier()
# fit the model
model.fit(X[train,:], Y_lab[train])

# get importance
importance = model.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()

Y_pred=model.predict(X[test])

classification_report(Y_lab[test], Y_pred)
metrics.accuracy_score(Y_lab[test], Y_pred)

tree.plot_tree(model) 


################ regression


reg_model = DecisionTreeRegressor()
reg_model.fit(X_inst[train_inst,:], Y_growth[train_inst])
Y_pred=reg_model.predict(X_inst[test_inst])

# get importance
importance = reg_model.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()


#########################


