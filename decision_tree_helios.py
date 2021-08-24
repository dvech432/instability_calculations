
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


filename = r'D:\Research\Data\DISP\NyKris_Durovcova_Helios.h5'
helios_df = pd.read_hdf(filename,key='/all data', start=0, stop=40000)

dim_params=np.array(helios_df[['B [nT]', 'n_p_core [cm^-3]','v_sw_core [m/s]','T_par_core [eV]','T_perp_core [eV]',
               'n_p_beam [cm^-3]','v_sw_beam [m/s]','T_par_beam [eV]','T_perp_beam [eV]',
               'n_alpha [cm^-3]','v_sw_alpha [m/s]','T_par_alpha [eV]','T_perp_alpha [eV]']])

cc = pd.read_csv(filename)

f=[0,1,3,4,5,6,7]
cc_x0=cc.columns
cc_col_names=cc_x0[f]
cc=np.array(cc)

at=np.where(cc[:,9]>0) # picking indeces of unstable cases

#### array with PLUME params for all cases
x_grid=np.where(cc[:,0]>0) # picking indeces all indeces
X_all=np.concatenate( (cc[x_grid,0], cc[x_grid,1], cc[x_grid,3], cc[x_grid,4], cc[x_grid,5], cc[x_grid,6], cc[x_grid,7]),axis=0).T #
Y_lab=np.array(pd.get_dummies( np.isnan(cc[:,9]) )[0]) # all labels

#### array with PLUME params for unstable cases
X_unstable=np.concatenate( (cc[at,0], cc[at,1], cc[at,3], cc[at,4], cc[at,5], cc[at,6], cc[at,7]),axis=0).T #

#### generating Y params for X_unstable
Y_theta=[]
for i in range(0,len(at[0])):
    Y_theta.append(math.atan2(cc[at[0][i],10],cc[at[0][i],11])*180/np.pi) # k-vector angle

Y_theta=np.reshape(np.array(Y_theta),(-1,1))
Y_mag = np.sqrt(cc[at,10]**2 + cc[at,11]**2).T # magnitude of the k-vector
Y_growth = cc[at,13].T # value of gamma

Y_all = np.concatenate([Y_theta.T, Y_mag.T, Y_growth.T],axis=0).T

###############################################
## split into train and test --- classification
from sklearn.model_selection import train_test_split
Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_all, Y_lab, test_size=0.2)

## split into train and test --- regression
Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_unstable, Y_all, test_size=0.2)

############## Classification
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

model = RandomForestClassifier() ### random forest led to the highest accuracy
# fit the model
model.fit(Xc_train, yc_train)

from sklearn.metrics import plot_confusion_matrix

disp = plot_confusion_matrix(model, Xc_test, yc_test, cmap=plt.cm.Blues)
plt.show()

############## speed test
# AA=[]
# r=1000
# for i in range(0,r):
#   AA.append(Xc_test)
# AA=np.reshape(AA,(3000*r,7))

# import time
# start_time = time.time()
# Y_pred=model.predict(AA)
# print("--- %s seconds ---" % (time.time() - start_time))
################
Y_pred=model.predict(Xc_test)
classification_report(yc_test, Y_pred)
metrics.accuracy_score(yc_test, Y_pred)

#### feature importance #1
from matplotlib import pyplot
plt.barh(cc_col_names, model.feature_importances_)
plt.xlabel("Random Forest Feature Importance")
plt.show()
#### feature importance #2
from sklearn.inspection import permutation_importance
perm_importance = permutation_importance(model, Xc_test, yc_test)
plt.barh(cc_col_names, perm_importance.importances_mean)
plt.xlabel("Permutation Importance")
plt.show()
#### feature importance #3
import shap
from shap import TreeExplainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(Xc_test)
plt.barh(cc_col_names, np.mean(np.abs(shap_values[0]),axis=0))
plt.xlabel("mean(|SHAP value|) (average impact on model output magnitude)")
plt.show()

###########################################################
###########################################################
################ regression

p=1 # 0: theta, 1: k magnitude, 2: growth rate
reg_model = RandomForestRegressor()
reg_model.fit(Xr_train, yr_train[:,p])
Y_pred=reg_model.predict(Xr_test)

#plt.scatter(yr_test[:,p],Y_pred)

from sklearn.metrics import mean_squared_error
errors = mean_squared_error(yr_test[:,p], Y_pred, squared=False)

errors_perc = np.std( 100*((yr_test[:,p]-Y_pred)/yr_test[:,p]) )

# report error
print(errors)
print(errors_perc)

# get importance
from matplotlib import pyplot
plt.barh(cc_col_names, model.feature_importances_)
plt.xlabel("Random Forest Feature Importance")
plt.show()
#### feature importance #2
from sklearn.inspection import permutation_importance
perm_importance = permutation_importance(model, Xc_test, yc_test)
plt.barh(cc_col_names, perm_importance.importances_mean)
plt.xlabel("Permutation Importance")
plt.show()
#### feature importance #3
import shap
from shap import TreeExplainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(Xc_test)
plt.barh(cc_col_names, np.mean(np.abs(shap_values[0]),axis=0))
plt.xlabel("mean(|SHAP value|) (average impact on model output magnitude)")
plt.show()

#########################

