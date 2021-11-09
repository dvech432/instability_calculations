########### build a classifier that identifies stable/unstable VDFs

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import pandas as pd
import numpy as np

## size of all the data:
RR0=0 # start id
RR=25000 # how many rows starting with RR0

##################### Load in image files
import os
folder_content=os.listdir(r'D:\Research\Data\DISP\VDF_images_helios_3comps_log')
#folder_content.sort()
import skimage.io as io
all_images = []
#for i in range(0,len(folder_content)):
for i in range(0,RR):    
  filename = r'D:\Research\Data\DISP\VDF_images_helios_3comps_log\\' + folder_content[i]
  img = io.imread(filename,as_gray=True)
  all_images.append(img)
  print(str(i))
all_images = np.array(all_images)   
all_images = all_images.reshape(-1, 100, 100, 1)
 
##################### Load in image labels

filename = r'D:\Research\Data\DISP\NyKris_Durovcova_Helios.h5'
helios_df = pd.read_hdf(filename,key='/all data', start=RR0, stop=RR0+RR)

label=np.array(helios_df[ 'gam_max_fine'])
label= np.nan_to_num(label, nan=0)
label[label>0]=1 # generate labels, 1== unstable, 0== stable

unstable_label=np.argwhere(label==1) # picking the unstable VDFs

# pol = np.array(helios_df[ ['Im(E_y)']    ])
# pol=pol[unstable_label[:,0]]
# pol[pol>0]=1
# pol[pol<0]=0

growth_params=np.array(helios_df[ ['kperp_max_fine', 'kpar_max_fine', 'gam_max_fine']    ])

mp=1.67272*10**-27 # proton mass
q=1.60217662*10**-19 # charge
kb=1.38064852*10**-23 # boltzmann constant
c=299792458 # speed of light
Omega=((10**-9)*q*helios_df['B [nT]'])/mp # cyclotron period

#growth_params[:,2]=growth_params[:,2]/Omega

Y = np.array( [np.sqrt(growth_params[:,0]**2 + growth_params[:,1]**2), growth_params[:,2]]).T

from scipy import stats

Y=np.log10(Y[unstable_label[:,0],:])

Y_mean=np.mean(Y,axis=0)
Y_std=np.mean(Y,axis=0)

Y_mean=Y.mean(axis=0) # saving mean
Y_std=Y.std(axis=0) # saving std
Y_norm=(Y-Y_mean)/Y_std # ready matrix
       
#Y_norm=stats.zscore(Y,axis=0)
#Y_norm=np.squeeze(Y_norm,axis=1)

all_images = all_images[unstable_label[:,0],:,:,:]


# %%    
##################### split into train and test
#### What parameter should be predicted? 0: |k|, 1: gamma/Omega
q=1

rr=len(unstable_label)

from sklearn.model_selection import train_test_split

idx_train, idx_test= train_test_split(np.arange(0,rr), test_size=0.15, random_state=42)

X_train=all_images[idx_train,:,:,:]
X_test=all_images[idx_test,:,:,:]
y_train=Y_norm[idx_train,q]
y_test=Y_norm[idx_test,q]

# pol_train=pol[idx_train]
# pol_test=pol[idx_test]

##################### model training

from cnn_regression import cnn_regression
m=cnn_regression()

# #from gen_AlexNet import gen_AlexNet
# #m=gen_AlexNet()

m.fit(X_train, y_train, validation_data=(X_test,y_test), batch_size=100,epochs=30, verbose=1)

#m.save(r'D:\Research\Data\Van_Allen_Probes\CNN\log_model_25k_q1') #alexnet

from keras.models import load_model
m = load_model(r'D:\Research\Data\Van_Allen_Probes\CNN\log_model_25k_q1')


from datetime import datetime
start_time = datetime.now()
pred=(m.predict(X_test))
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))


import matplotlib.pyplot as plt
#plt.scatter(y_test,pred)
#plt.scatter((Y_mean[q]+(y_test*Y_std[q])),(Y_mean[q]+(pred*Y_std[q])))
plt.hist2d((Y_mean[q]+(y_test*Y_std[q])).flatten(),(Y_mean[q]+(pred*Y_std[q])).flatten(), bins=50, cmap='Blues')
cb = plt.colorbar()
cb.set_label('counts in bin')
if q==1:
  plt.plot(np.linspace(-3,0),np.linspace(-3,0),c='black',linewidth=0.5)
  plt.xlim((-3,-0.5))
  plt.ylim((-3,-0.5))
if q==0:
  plt.plot(np.linspace(-0.5,0),np.linspace(-0.5,0),c='black',linewidth=0.5)
  plt.xlim((-0.5,0))
  plt.ylim((-0.5,0))
  
plt.xlabel('True')
plt.ylabel('Predicted')
#m.save(r'D:\Research\Data\Van_Allen_Probes\CNN\classification_trained') #alexnet
#from keras.models import load_model
#m = load_model(r'D:\Research\Data\Van_Allen_Probes\CNN\classification_trained') #alexnet

# %% kernel density estimation
# from scipy.stats import gaussian_kde
# data = np.vstack([(Y_mean[q]+(y_test*Y_std[q])).flatten(), (Y_mean[q]+(pred*Y_std[q])).flatten()])
# kde = gaussian_kde(data)

# # evaluate on a regular grid
# xgrid = np.linspace(-3.5,0, 80)
# ygrid = np.linspace(-3.5, 0, 80)
# Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
# Z = kde.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))

# # Plot the result as an image
# plt.imshow(Z.reshape(Xgrid.shape),
#            origin='lower', aspect='auto',
#            extent=[-3.5, 3.5, -6, 6],
#            cmap='Blues')
# cb = plt.colorbar()
# cb.set_label("density")
# plt.plot(np.linspace(-4,0),np.linspace(-4,0),c='black')
# %% predicting polarization

# from cnn_classification import cnn_classification
# m=cnn_classification()

# # #from gen_AlexNet import gen_AlexNet
# # #m=gen_AlexNet()

# m.fit(X_train, pol_train, validation_data=(X_test,pol_test), batch_size=100,epochs=60, verbose=1)

# %%
### grid search
import pickle
from cnn_regression_grid_search import cnn_regression_grid_search

lr=np.logspace(-6,-1,6)
dc=np.logspace(-6,-1,6)
op_type=['Adam','SGD','RMSprop']

u=0
for k in range(0,len(op_type)):
  params=[]
  train_history=[]  
  for i in range(0,len(lr)):
    for j in range(0,len(dc)):
      m=cnn_regression_grid_search(lr[i], dc[j],op_type[k])
      seq_model=m.fit(X_train, y_train, validation_data=(X_test,y_test), batch_size=100,epochs=15, verbose=1)
            
      params.append([op_type[k],lr[i],dc[j]])
      train_history.append(seq_model.history)
      print(str(u))
      u=u+1
      
  filename=(r'D:\Research\Data\Van_Allen_Probes\grid_search_regression\\' + op_type[k] + '.sav')
  pickle.dump([train_history,params], open(filename, 'wb'))      

# %%
### plotting the result of the grid search
k=0
filename=(r'D:\Research\Data\Van_Allen_Probes\grid_search_regression\\' + op_type[k] + '.sav')
results = pickle.load(open(filename, 'rb'))
  
val_acc=[]
x=[]
y=[]
slope=[]
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

slope=np.array(slope)
val_acc=np.array(val_acc)
x=np.array(x)
y=np.array(y)  

print('Best learning rate:'  + str(x[np.argmax(val_acc)]))
print('Best decay rate:'  + str(y[np.argmax(val_acc)]))
print('Peak accuracy:'  + str(val_acc[np.argmax(val_acc)]))

#tp_param=np.nanmean(plume_array[tp,:],axis=0).T
#fp_param=np.nanmean(plume_array[fp,:],axis=0).T
#print(tp_param/fp_param)

# %% understanding the properties of correctly and incorrectly classified cases

dim_params=np.array(helios_df[['B [nT]', 'n_p_core [cm^-3]','v_sw_core [m/s]','T_par_core [eV]','T_perp_core [eV]',
               'n_p_beam [cm^-3]','v_sw_beam [m/s]','T_par_beam [eV]','T_perp_beam [eV]',
               'n_alpha [cm^-3]','v_sw_alpha [m/s]','T_par_alpha [eV]','T_perp_alpha [eV]']])

dim_params=dim_params[idx_test,:]

growth_params=np.array(helios_df[ ['kperp_max_fine', 'kpar_max_fine', 'gam_max_fine']    ])
growth_params=growth_params[idx_test,:]

mp=1.67272*10**-27 # proton mass
q=1.60217662*10**-19 # charge
kb=1.38064852*10**-23 # boltzmann constant
c=299792458 # speed of light

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
import matplotlib.pyplot as plt
fn=np.argwhere( (y_test==1) & (pred_labels==0))
fn=fn[:,0]

plt.hist(np.log10(growth_params[:,2]),density=False ,edgecolor='None', alpha = 0.5, bins=20 )
plt.hist(np.log10(growth_params[fn,2]),density=False,edgecolor='None', alpha = 0.5, bins=20 )
plt.xlabel('gamma')


for i in range(0,12):
  plt.hist(dim_params[:,i],bins=20,density=True,edgecolor='None', alpha = 0.5,)
  plt.hist(dim_params[fn,i],bins=20,density=True,edgecolor='None', alpha = 0.5,)
  plt.title('Column: ' + str(i) )
  plt.show()
  
  
plt.hist( (dim_params[:,6]-dim_params[:,2])/1000 ,density=True ,edgecolor='None', alpha = 0.5, bins=20 )
plt.hist( (dim_params[fn,6]-dim_params[fn,2])/1000 ,density=True,edgecolor='None', alpha = 0.5, bins=20 )
plt.xlabel('v beam - v core')  

plt.hist( (dim_params[:,10]-dim_params[:,2])/1000 ,density=True ,edgecolor='None', alpha = 0.5, bins=20 )
plt.hist( (dim_params[fn,10]-dim_params[fn,2])/1000 ,density=True,edgecolor='None', alpha = 0.5, bins=20 )
plt.xlabel('v alpha - v core')

# %% checking what particle population combination can be predicted with lowest accuracy

p1=np.argwhere( (np.isnan(dim_params[:,6])==False) & (np.isnan(dim_params[:,12])==False))
p2=np.argwhere( (np.isnan(dim_params[:,6])==True) & (np.isnan(dim_params[:,12])==False))
p3=np.argwhere( (np.isnan(dim_params[:,6])==False) & (np.isnan(dim_params[:,12])==True))
p4=np.argwhere( (np.isnan(dim_params[:,6])==True) & (np.isnan(dim_params[:,12])==True))

print(len(np.argwhere(pred_labels[p1]==y_test[p1]))/len(p1))
print(len(np.argwhere(pred_labels[p2]==y_test[p2]))/len(p2))
print(len(np.argwhere(pred_labels[p3]==y_test[p3]))/len(p3))
print(len(np.argwhere(pred_labels[p4]==y_test[p4]))/len(p4))

# %% generating figures for paper

# %% Figure 5 histogram of true vs predicted values
# q==1
h=np.arange(0,101,10)
import matplotlib.pyplot as plt
#plt.scatter(y_test,pred)
#plt.scatter((Y_mean[q]+(y_test*Y_std[q])),(Y_mean[q]+(pred*Y_std[q])))
plt.figure(dpi=550)

xbins = np.logspace(-4,0,40)
ybins = np.logspace(-4,0,40)

plt.hist2d(10**(Y_mean[q]+(y_test*Y_std[q])).flatten(),10**(Y_mean[q]+(pred*Y_std[q])).flatten(),cmap='viridis',bins=(xbins,ybins) )
plt.xscale('log')
plt.yscale('log')  

cb = plt.colorbar()
cb.set_label('counts in bin')
plt.plot(np.logspace(-4,0),np.logspace(-4,0),c='black',linewidth=1)

plt.xlabel('Ground truth '+ '$\gamma/\Omega_p$')
plt.ylabel('Predicted ' + '$\gamma/\Omega_p$')
#plt.xlim((-3,-0.5))
#plt.ylim((-3,-0.5))
plt.savefig(r'C:\Users\vechd\.spyder-py3\instability_calc\Figures\reg_gamma.jpg', format='jpg')

#################### #########################

plt.figure(dpi=750)
h=np.arange(0,101,10)
xx=10**(Y_mean[q]+(y_test*Y_std[q])).flatten()
yy=10**(Y_mean[q]+(pred*Y_std[q])).flatten()

avg=[]
err=[]
for i in range(len(h)-1):
  h0=np.nonzero( (xx > np.percentile(xx,h[i] ) ) & (xx < np.percentile(xx,h[i+1] ) )  )[0]
  avg=np.append(avg,np.nanmean(xx[h0]))
  err=np.append(err, np.std( (np.abs( (xx[h0]-yy[h0]) )    )/xx[h0] ) )


plt.plot(np.logspace(-3,-1), np.full((50, 1), 1),c='black')
plt.plot(avg, err*100)
plt.scatter(avg, err*100)
plt.xscale('log')
plt.ylim((0,5*100))
plt.xlim((10**-3,10**-1))
plt.xlabel('Ground truth '+ '$\gamma/\Omega_p$')
plt.ylabel('1$\sigma$ relative error')

plt.savefig(r'C:\Users\vechd\.spyder-py3\instability_calc\Figures\reg_error.jpg', format='jpg')


package=[]
package.append(avg)
package.append(err)
from scipy.io import savemat
filename=(r'D:\Research\Data\Van_Allen_Probes\q1100.mat')
savemat(filename, {"foo":package})


# %% q=0
h=np.arange(0,101,10)
import matplotlib.pyplot as plt
plt.figure(dpi=550)
plt.hist2d(10**(Y_mean[q]+(y_test*Y_std[q])).flatten(),10**(Y_mean[q]+(pred*Y_std[q])).flatten(), bins=75, cmap='viridis') 
cb = plt.colorbar()
cb.set_label('counts in bin')
plt.plot(np.linspace(0,1),np.linspace(0,1),c='black',linewidth=1)
plt.xlim((2*10**-1,1))
plt.ylim((2*10**-1,1))
plt.xscale('log')
plt.yscale('log')  
plt.xlabel('Ground truth '+ r'$k\rho_p$')
plt.ylabel('Predicted ' + r'$k\rho_p$')


plt.savefig(r'C:\Users\vechd\.spyder-py3\instability_calc\Figures\reg_rho.jpg', format='jpg')


#################### #########################
# analyzing relative error

plt.figure(dpi=750)
h=np.arange(0,101,10)
xx=10**(Y_mean[q]+(y_test*Y_std[q])).flatten()
yy=10**(Y_mean[q]+(pred*Y_std[q])).flatten()

avg=[]
err=[]
for i in range(len(h)-1):
  h0=np.nonzero( (xx > np.percentile(xx,h[i] ) ) & (xx < np.percentile(xx,h[i+1] ) )  )[0]
  avg=np.append(avg,np.nanmean(xx[h0]))
  err=np.append(err, np.std( (np.abs( (xx[h0]-yy[h0]) )    )/xx[h0] ) )


plt.plot(np.logspace(-3,-1), np.full((50, 1), 1),c='black')
plt.plot(avg, err*100)
plt.scatter(avg, err*100)
plt.xscale('log')
plt.ylim((0,0.5*100))
plt.xlim((2*10**-1,1))
plt.xlabel('Ground truth '+ r'$k\rho_p$')
plt.ylabel('1$\sigma$ relative error [%]')

plt.savefig(r'C:\Users\vechd\.spyder-py3\instability_calc\Figures\reg_rho_error.jpg', format='jpg')


package=[]
package.append(avg)
package.append(err)
from scipy.io import savemat
filename=(r'D:\Research\Data\Van_Allen_Probes\q0100.mat')
savemat(filename, {"foo":package})

# %% make final plots
import scipy.io as sio
plt.figure(dpi=750)
filename=(r'D:\Research\Data\Van_Allen_Probes\q0100.mat')
package=sio.loadmat(filename)
plt.plot(package['foo'][0], package['foo'][1]*100,label='100x100')
plt.scatter(package['foo'][0], package['foo'][1]*100)

filename=(r'D:\Research\Data\Van_Allen_Probes\q035.mat')
package=sio.loadmat(filename)
plt.plot(package['foo'][0], package['foo'][1]*100,label='35x35')
plt.scatter(package['foo'][0], package['foo'][1]*100)


filename=(r'D:\Research\Data\Van_Allen_Probes\q025.mat')
package=sio.loadmat(filename)
plt.plot(package['foo'][0], package['foo'][1]*100,label='25x25')
plt.scatter(package['foo'][0], package['foo'][1]*100)

plt.legend()
plt.xscale('log')
plt.ylim((0,0.5*100))
plt.xlim((2*10**-1,1))
plt.xlabel('Ground truth '+ r'$k\rho_p$')
plt.ylabel('1$\sigma$ relative error [%]')

plt.savefig(r'C:\Users\vechd\.spyder-py3\instability_calc\Figures\reg_rho_errors.jpg', format='jpg')


import scipy.io as sio
plt.figure(dpi=750)
filename=(r'D:\Research\Data\Van_Allen_Probes\q1100.mat')
package=sio.loadmat(filename)
plt.plot(package['foo'][0], package['foo'][1]*100,label='100x100')
plt.scatter(package['foo'][0], package['foo'][1]*100)

filename=(r'D:\Research\Data\Van_Allen_Probes\q135.mat')
package=sio.loadmat(filename)
plt.plot(package['foo'][0], package['foo'][1]*100,label='35x35')
plt.scatter(package['foo'][0], package['foo'][1]*100)


filename=(r'D:\Research\Data\Van_Allen_Probes\q125.mat')
package=sio.loadmat(filename)
plt.plot(package['foo'][0], package['foo'][1]*100,label='25x25')
plt.scatter(package['foo'][0], package['foo'][1]*100)

plt.legend()
plt.xscale('log')
plt.ylim((0,5*100))
plt.xlim((10**-3,10**-1))
plt.xlabel('Ground truth '+ '$\gamma/\Omega_p$')
plt.ylabel('1$\sigma$ relative error [%]')

plt.savefig(r'C:\Users\vechd\.spyder-py3\instability_calc\Figures\reg_omega_errors.jpg', format='jpg')


