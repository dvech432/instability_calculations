


##################### generating proton core, beam, alpha with random values
import numpy as np

S=10000
a = 400*1000
b = 600*1000
v_core = np.random.uniform(a,b,S)

a = 10*1000
b = 120*1000
v_beam = v_core+ np.random.uniform(a,b,S)

a = 50*1000
b = 150*1000
v_a = v_core + np.random.uniform(a,b,S)

a = 20*1000
b = 80*1000
vth_core = np.random.uniform(a,b,S)

a = 30*1000
b = 90*1000
vth_beam = np.random.uniform(a,b,S)

a = 30*1000;
b = 90*1000;
vth_a = np.random.uniform(a,b,S)

a = 1
b = 100
n_core = np.random.uniform(a,b,S)

a = 0.05
b = 1
n_beam = n_core*np.random.uniform(a,b,S)

a = 0.01
b = 0.05
n_a = (n_core+n_beam)*np.random.uniform(a,b,S)

V=np.logspace(np.log10(300),np.log10(5000),59);
v = (np.sqrt(2*1.60217662*10**-19*(V)/(1.6726219*10**-27)))

V_high=np.logspace(np.log10(307.317073),np.log10(5121.95121),59)
V_low=np.logspace(np.log10(292.68292),np.log10(4878.04878),59)

dV=(V_high-V_low)

m = 1.67272*10**-27 # proton mass
q=1.60217662*10**-19 # charge

#G = normrnd(0,1,[1000000 1]); # add random noise to the VDF
################ plotting
import matplotlib as plt
from get_fv import get_fv
i=3
f_core = get_fv(v_core[i],vth_core[i], dV,q,m,v,n_core[i])
f_beam = get_fv(v_beam[i],vth_beam[i], dV,q,m,v,n_beam[i])   
f_alpha = get_fv(v_a[i],vth_a[i], dV,q,m,v,n_a[i])   

#plt.pyplot.scatter((v/1000)-(v_core[i]/1000),np.log10(f_core))

plt.pyplot.scatter((v/1000),np.log10(f_core))

#############
#############
###### fixing the rotation bug:
    
y0=np.zeros(np.reshape(v,(-1,1)).shape)
x0=np.reshape( (v/1000), (-1,1))
    
point= np.concatenate((x0,y0),axis=1)
point[:,0]=point[:,0]-v_core[i]/1000

rot_coord=np.empty((0,2))
rot_psd=np.empty((0,1))
angle=np.linspace(0,180,18)

from rotate_v2 import rotate_v2
for i in range(0,len(angle)):
  rotated_coord=rotate_v2(point, origin=(0,0), degrees=angle[i])
  rot_coord = np.concatenate([rot_coord, rotated_coord])
  rot_psd = np.concatenate([rot_psd,np.log10(np.reshape(f_core,(-1,1)))])

import matplotlib as plt  
plt.pyplot.scatter(rot_coord[:,0],rot_coord[:,1])
######## trisurf method for plotting mesh grid

coords=np.concatenate((rot_coord,rot_psd),axis=1)
coords2=np.delete(coords,np.where(coords[:,2]<0),axis=0)

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Make the plot
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_trisurf(coords2[:,0]+v_core[i]/1000, coords2[:,1], coords2[:,2], cmap=plt.cm.jet, linewidth=0.2)
#plt.show()
 
ax.view_init(90, 90)
plt.show()


##### meshgrid like approach

X,Y=np.meshgrid(np.logspace(np.log10(300),np.log10(5000),59),np.logspace(np.log10(300),np.log10(5000),59))

Z=np.sqrt(X**2 + Y**2)


############## interpolate the generated VDF to a linear velocity grid and then rotate it


