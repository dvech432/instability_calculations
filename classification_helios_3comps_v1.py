########### build a classifier that identifies stable/unstable VDFs

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import pandas as pd
import numpy as np

## size of all the data:
RR=25000

##################### Load in image files
import os
folder_content=os.listdir(r'D:\Research\Data\DISP\VDF_images_helios_3comps_log35')
#folder_content.sort()
import skimage.io as io
all_images = []
#for i in range(0,len(folder_content)):
for i in range(0,RR):    
  filename = r'D:\Research\Data\DISP\VDF_images_helios_3comps_log35\\' + folder_content[i]
  img = io.imread(filename,as_gray=True)
  all_images.append(img)
  print(str(i))
all_images = np.array(all_images)   
all_images = all_images.reshape(-1, 100, 100, 1)
 
##################### Load in image labels

filename = r'D:\Research\Data\DISP\NyKris_Durovcova_Helios.h5'
helios_df = pd.read_hdf(filename,key='/all data', start=0, stop=RR)

label=np.array(helios_df[ 'gam_max_fine'])
label= np.nan_to_num(label, nan=0)
label[label>0]=1

### new idea: exclude marginally unstable cases
#label=pd.DataFrame(pd.get_dummies(plume[plume.columns[13]]>10**-2.5)  )

### old method
label=pd.DataFrame(pd.get_dummies((label==1)))

label=np.array(label.drop(0,axis=1))

    
##################### split into train and test
rr=RR

from sklearn.model_selection import train_test_split

idx_train, idx_test= train_test_split(np.arange(0,rr), test_size=0.15, random_state=42)

X_train=all_images[idx_train,:,:,:]
X_test=all_images[idx_test,:,:,:]
y_train=label[idx_train]
y_test=label[idx_test]

del all_images
##################### model training

from cnn_classification import cnn_classification
m=cnn_classification()

#from gen_AlexNet import gen_AlexNet
#m=gen_AlexNet()

m.fit(X_train, y_train, validation_data=(X_test,y_test), batch_size=100,epochs=30, verbose=1)

pred_labels=np.round(m.predict(X_test))
pred_score=(m.predict(X_test))

m.save(r'D:\Research\Data\Van_Allen_Probes\CNN\log_model_25k') #alexnet

#####  ######  ######  ###### #####
from keras.models import load_model
m = load_model(r'D:\Research\Data\Van_Allen_Probes\CNN\log_model_25k')

pred_labels=np.round(m.predict(X_test))
pred_score=(m.predict(X_test))

classification=[]
classification.append(y_test)
classification.append(pred_labels)
from scipy.io import savemat
filename=(r'D:\Research\Data\Van_Allen_Probes\classification35.mat')
savemat(filename, {"foo":classification})


# %% estimation of execution time
from dnn_profiler import dnn_profiler
import time
s0=time.process_time()
dnn_profiler(m,X_test)
s1=time.process_time()

print('Total CPU time')
print(s1-s0)
print('CPU time per execution')
print((s1-s0)/len(idx_test))
# %%
### grid search, 20k sample size should be used
import pickle
from cnn_classification_grid_search import cnn_classification_grid_search

lr=np.logspace(-6,-1,6)
dc=np.logspace(-6,-1,6)
op_type=['Adam','SGD','RMSprop']

u=0
for k in range(0,len(op_type)):
  params=[]
  train_history=[]  
  for i in range(0,len(lr)):
    for j in range(0,len(dc)):
      m=cnn_classification_grid_search(lr[i], dc[j],op_type[k])
      seq_model=m.fit(X_train, y_train, validation_data=(X_test,y_test), batch_size=100,epochs=5, verbose=1)
            
      params.append([op_type[k],lr[i],dc[j]])
      train_history.append(seq_model.history)
      print(str(u))
      u=u+1
      
  filename=(r'D:\Research\Data\Van_Allen_Probes\grid_search_log\\' + op_type[k] + '.sav')
  pickle.dump([train_history,params], open(filename, 'wb'))      

# %%
### plotting the result of the grid search
import pickle
k=2
op_type=['Adam','SGD','RMSprop']
filename=(r'D:\Research\Data\Van_Allen_Probes\grid_search_log\\' + op_type[k] + '.sav')
results = pickle.load(open(filename, 'rb'))
  
val_acc=[]
x=[]
y=[]
slope=[]
gradient=[]
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


for i in range(0,len(results[0])):
  model = LinearRegression()
  model = LinearRegression().fit(np.reshape(np.linspace(0,5,5),(-1,1)), np.reshape(results[0][i]['val_accuracy'][-5:],(-1,1)))
  slope.append(model.coef_[0][0])
  plt.plot(results[0][i]['val_accuracy'])
  val_acc.append(results[0][i]['val_accuracy'][-1])
  x.append(results[1][i][1]) # learning rate
  y.append(results[1][i][2]) # decay rate
  gradient.append(np.diff((results[0][i]['val_accuracy'][-2:])))
  
slope=np.array(slope)
val_acc=np.array(val_acc)
x=np.array(x)
y=np.array(y)  
gradient=np.array(gradient)

print('Best learning rate:'  + str(x[np.argmax(val_acc)]))
print('Best decay rate:'  + str(y[np.argmax(val_acc)]))
print('Peak accuracy:'  + str(val_acc[np.argmax(val_acc)]))


plt.scatter(gradient,val_acc)
# plt.scatter(val_acc,slope)
# plt.xlim((0.7,0.85))
# plt.ylim((-0.01,0.01))
# %% ############ plotting confusion matrix

#from sklearn.metrics import classification_report
#classification_report(y_test,pred_labels)

tp=np.where( (pred_labels==1) & (y_test==1) )[0]
fp=np.where( (pred_labels==1) & (y_test==0) )[0]

tn=np.where( (pred_labels==0) & (y_test==0) )[0]
fn=np.where( (pred_labels==0) & (y_test==1) )[0]

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt     

labels = ['Stable', 'Unstable']
cm = confusion_matrix(y_test, pred_labels)

# ax= plt.subplot()
# sns.heatmap(cm, annot=True, ax = ax,cmap="YlGnBu"); #annot=True to annotate cells

# # labels, title and ticks
# ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
# ax.set_title('Confusion Matrix'); 
# ax.xaxis.set_ticklabels(['Stable', 'Unstable']); ax.yaxis.set_ticklabels(['Stable', 'Unstable']);

from plot_conf_matrix import pretty_plot_confusion_matrix

array = cm
    #get pandas dataframe
df_cm = pd.DataFrame(array, index=range(1,3), columns=range(1,3))
    #colormap: see this and choose your more dear
cmap = 'YlGnBu'
pretty_plot_confusion_matrix(df_cm, cmap=cmap)
# %% understanding the properties of correctly and incorrectly classified cases

dim_params=np.array(helios_df[['B [nT]', 'n_p_core [cm^-3]','v_sw_core [m/s]','T_par_core [eV]','T_perp_core [eV]',
               'n_p_beam [cm^-3]','v_sw_beam [m/s]','T_par_beam [eV]','T_perp_beam [eV]',
               'n_alpha [cm^-3]','v_sw_alpha [m/s]','T_par_alpha [eV]','T_perp_alpha [eV]']])

dim_params=dim_params[idx_test,:]

growth_params=np.array(helios_df[ ['kperp_max_fine', 'kpar_max_fine', 'gam_max_fine']    ])
growth_params=growth_params[idx_test,:]

growth_params=np.append(growth_params,np.reshape(np.sqrt(growth_params[:,0]**2+growth_params[:,1]**2),(-1,1)),axis=1)

mp=1.67272*10**-27 # proton mass
q=1.60217662*10**-19 # charge
kb=1.38064852*10**-23 # boltzmann constant
c=299792458 # speed of light
Omega=((10**-9)*q*helios_df['B [nT]'])/mp # cyclotron period
Omega=Omega[idx_test]
growth_params[:,2]=growth_params[:,2]/Omega
##### eV to Kelvin conversion
dim_params[:,4]=11604.525*dim_params[:,4]
dim_params[:,3]=11604.525*dim_params[:,3]
dim_params[:,8]=11604.525*dim_params[:,8]
dim_params[:,7]=11604.525*dim_params[:,7]
dim_params[:,11]=11604.525*dim_params[:,11]
dim_params[:,12]=11604.525*dim_params[:,12]

#### VDF params

v_core=dim_params[:,2]
v_beam=dim_params[:,6]
v_alpha=dim_params[:,10]

n_core=dim_params[:,1]
n_beam=dim_params[:,5]
n_alpha=dim_params[:,9]

T_core_perp=dim_params[:,4]
T_core_par=dim_params[:,3]

vth_core_perp=(2*kb*T_core_perp/mp)**(0.5)
vth_core_par=(2*kb*T_core_par/mp)**(0.5)

T_beam_perp=dim_params[:,8]
T_beam_par=dim_params[:,7]

vth_beam_perp=(2*kb*T_beam_perp/mp)**(0.5)
vth_beam_par=(2*kb*T_beam_par/mp)**(0.5)

T_alpha_perp=dim_params[:,12]
T_alpha_par=dim_params[:,11]

vth_alpha_perp=(2*kb*T_beam_perp/(4*mp))**(0.5)
vth_alpha_par=(2*kb*T_beam_par/(4*mp))**(0.5)

###### B-field
b0 = dim_params[:,0]
n_total=np.nansum(np.dstack((n_core,n_beam,n_alpha)),2)
alfv = (b0*10**-9./np.sqrt(4*np.pi*10**-7.*(n_total)*100**3*1.67*10**-27))

# %%
# import matplotlib.pyplot as plt
# fn=np.argwhere( (y_test==1) & (pred_labels==0))
# fn=fn[:,0]

# plt.hist(np.log10(growth_params[:,2]) ,density=True,edgecolor='None', alpha = 0.5, bins=20 )
# plt.hist(np.log10(growth_params[fn,2]), density=True, edgecolor='None', alpha = 0.5, bins=20 )
# plt.xlabel('gamma/Omega')


# for i in range(0,12):
#   plt.hist(dim_params[:,i],bins=20,density=True,edgecolor='None', alpha = 0.5,)
#   plt.hist(dim_params[fn,i],bins=20,density=True,edgecolor='None', alpha = 0.5,)
#   plt.title('Column: ' + str(i) )
#   plt.show()
  
  
# plt.hist( (dim_params[:,6]-dim_params[:,2])/1000 ,density=True ,edgecolor='None', alpha = 0.5, bins=20 )
# plt.hist( (dim_params[fn,6]-dim_params[fn,2])/1000 ,density=True,edgecolor='None', alpha = 0.5, bins=20 )
# plt.xlabel('v beam - v core')  

# plt.hist( (dim_params[:,10]-dim_params[:,2])/1000 ,density=True ,edgecolor='None', alpha = 0.5, bins=20 )
# plt.hist( (dim_params[fn,10]-dim_params[fn,2])/1000 ,density=True,edgecolor='None', alpha = 0.5, bins=20 )
# plt.xlabel('v alpha - v core')

# %% checking what particle population combination can be predicted with lowest accuracy

# core only
p1=np.argwhere( (np.isnan(dim_params[:,6])==False) & (np.isnan(dim_params[:,12])==False))
# core + beam
p2=np.argwhere( (np.isnan(dim_params[:,6])==True) & (np.isnan(dim_params[:,12])==False))
# core + alpha
p3=np.argwhere( (np.isnan(dim_params[:,6])==False) & (np.isnan(dim_params[:,12])==True))
# core + beam + alpha
p4=np.argwhere( (np.isnan(dim_params[:,6])==True) & (np.isnan(dim_params[:,12])==True))

print(len(np.argwhere(pred_labels[p1]==y_test[p1]))/len(p1))
print(len(np.argwhere(pred_labels[p2]==y_test[p2]))/len(p2))
print(len(np.argwhere(pred_labels[p3]==y_test[p3]))/len(p3))
print(len(np.argwhere(pred_labels[p4]==y_test[p4]))/len(p4))


print(len(np.argwhere(pred_labels[p1]==y_test[p1]))/len(p1))
print(len(np.argwhere(pred_labels[p2]==y_test[p2]))/len(p2))
print(len(np.argwhere(pred_labels[p3]==y_test[p3]))/len(p3))
print(len(np.argwhere(pred_labels[p4]==y_test[p4]))/len(p4))

# %% Figure 1 in the paper - a random VDF
v_xa, v_ya= np.meshgrid(np.linspace(-2,2,94),np.linspace(-2,2,94))

from scipy.interpolate import griddata

points = np.array( (v_xa.flatten(), v_ya.flatten()) ).T

import os
folder_content=os.listdir(r'D:\Research\Data\DISP\VDF_images_helios_3comps_log')
#folder_content.sort()
import skimage.io as io
all_images = []
#for i in range(0,len(folder_content)):
for i in range(0,10):    
  filename = r'D:\Research\Data\DISP\VDF_images_helios_3comps_log\\' + folder_content[i]
  img = io.imread(filename,as_gray=True)
  all_images.append(img)
  print(str(i))
all_images = np.array(all_images)   
all_images = all_images.reshape(-1, 100, 100, 1)

values = np.squeeze(all_images[3,3:-3,3:-3,:],axis=2)/np.max(np.squeeze(all_images[3,3:-3,3:-3,:],axis=2))
    
fig = plt.figure(dpi=550)

ax = fig.add_axes([0, 0, 1, 1])
    
cnt = ax.pcolor(v_xa,v_ya, values, cmap='viridis',shading='auto') # Alfv norm
    #cnt = ax.pcolor(v_x,v_y,np.log10( (Z_core+Z_beam)/np.max((Z_core+Z_beam))    ), cmap='viridis') # Thermal speed norm
  
#ax.axis('off')
ax.set_aspect('equal', 'box')
fig.colorbar(cnt)
plt.ylabel('$V_{\perp}/V_A$')
plt.xlabel('$V_{||}/V_A$')

### save high quality figure
fig.savefig(r'C:\Users\vechd\.spyder-py3\instability_calc\Figures\vdf_example.jpg', format='jpg',bbox_inches = "tight")


# %% Figure 2 in the paper - confusion matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, pred_labels))

target_names = ['Stable', 'Unstable']
print(classification_report(y_test, pred_labels, target_names=target_names))

tp=np.where( (pred_labels==1) & (y_test==1) )[0]
fp=np.where( (pred_labels==1) & (y_test==0) )[0]

tn=np.where( (pred_labels==0) & (y_test==0) )[0]
fn=np.where( (pred_labels==0) & (y_test==1) )[0]

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt     

labels = ['Stable', 'Unstable']
cm = confusion_matrix(y_test, pred_labels)

from plot_confusion_matrix import plot_confusion_matrix

plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True)

sensitivity1 = cm[0,0]/(cm[0,0]+cm[0,1])
print('Sensitivity : ', sensitivity1 )

specificity1 = cm[1,1]/(cm[1,0]+cm[1,1])
print('Specificity : ', specificity1)

#### checking the true positive rate for specific subset:
  
print(len(np.where( (pred_labels==0) & (np.reshape(growth_params[:,3],(-1,1))>0.4) )[0])/len(np.where( (growth_params[:,3]>0.4) )[0]))

# %% Figure 3 in the paper - histogram with misclassification

import matplotlib.pyplot as plt
fn=np.argwhere( (y_test==1) & (pred_labels==0))
fn=fn[:,0]


############## gamme/Omega plot
bins = np.logspace(-4,0,30)
plt.figure(dpi=550)
plt.hist((growth_params[:,2]) , density=False,edgecolor='None', alpha = 0.9, bins=bins ,label='All test data')
plt.hist((growth_params[fn,2]), density=False, edgecolor='None', alpha = 0.9, bins=bins,label='False negative' )
plt.xlabel('$\gamma/\Omega_p$')
plt.ylabel('Number of occurrence')
plt.xscale('log')
plt.legend()

plt.savefig(r'C:\Users\vechd\.spyder-py3\instability_calc\Figures\hist_gamma.jpg', format='jpg')

############## |k*rho| plot
bins = np.logspace(-1,0.3,30)
plt.figure(dpi=550)
plt.hist((growth_params[:,3]) , density=False,edgecolor='None', alpha = 0.9, bins=bins ,label='All test data')
plt.hist((growth_params[fn,3]), density=False, edgecolor='None', alpha = 0.9, bins=bins,label='False negative' )
plt.xlabel(r'$k\rho_p$')
plt.ylabel('Number of occurrence')
plt.xscale('log')
plt.legend()

plt.savefig(r'C:\Users\vechd\.spyder-py3\instability_calc\Figures\hist_k.jpg', format='jpg')


