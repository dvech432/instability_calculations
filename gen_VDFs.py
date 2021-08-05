##### generating images of the VDFs, using dim_params.txt
import numpy as np
import pandas as pd

#### read in params
filename = r'C:\Users\vechd\.spyder-py3\instability_calc\dim_params.txt'
dim_params=np.array(pd.read_csv(filename))

m=1.67272*10**-27 # proton mass
q=1.60217662*10**-19 # charge
kb=1.38064852*10**-23 # boltzmann constant
c=299792458 # speed of light
#### VDF params

v_core=dim_params[:,2]
v_beam=dim_params[:,6]

n_core=dim_params[:,1]
n_beam=dim_params[:,5]

T_core_perp=dim_params[:,4]
T_core_par=dim_params[:,3]

vth_core_perp=(2*kb*T_core_perp/m)**(0.5)
vth_core_par=(2*kb*T_core_par/m)**(0.5)

T_beam_perp=dim_params[:,8]
T_beam_par=dim_params[:,7]

vth_beam_perp=(2*kb*T_beam_perp/m)**(0.5)
vth_beam_par=(2*kb*T_beam_par/m)**(0.5)

###### B-field
b0 = dim_params[:,0]
alfv = (b0*10**-9./np.sqrt(4*np.pi*10**-7.*(n_core+n_beam)*100**3*1.67*10**-27))

v_x, v_y=np.meshgrid(np.linspace(-200*1000,200*1000,100),np.linspace(-200*1000,200*1000,100))
v_xa, v_ya= np.meshgrid(np.linspace(-3,3,100),np.linspace(-3,3,100))

import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import gc

from PIL import Image

for i in range(0,len(n_core)):
  Z_core=n_core[i]*(m/(2*np.pi*kb*T_core_perp[i]))*(m/(2*np.pi*kb*T_core_perp[i]))**(0.5)*np.exp(-m*((v_x)**2)/(2*kb*T_core_perp[i]))*np.exp(-m*((v_y)**2)/(2*kb*T_core_par[i]))  
  Z_beam=n_beam[i]*(m/(2*np.pi*kb*T_beam_perp[i]))*(m/(2*np.pi*kb*T_beam_perp[i]))**(0.5)*np.exp(-m*(((v_beam[i]-v_core[i])-v_x)**2)/(2*kb*T_beam_perp[i]))*np.exp(-m*((v_y)**2)/(2*kb*T_beam_par[i]))
  ### interpolation to the Alfv normalized grid
  v_xn= v_x/alfv[i]
  v_yn=v_y/alfv[i]
  points = np.array( (v_xn.flatten(), v_yn.flatten()) ).T
  values = ((Z_core.flatten()+Z_beam.flatten())/np.max(Z_core.flatten()+Z_beam.flatten()))
  grid_z0 = griddata(points, values, (v_xa, v_ya), method='linear')
  grid_z0 = np.nan_to_num(grid_z0, nan=0.000001)
  
  fig = plt.figure()
  ax = fig.add_axes([0, 0, 1, 1])
  
  
  ###### Contour approach
  levels=np.logspace(-1,0,6)
  #cnt = ax.contour(v_xa,v_ya, grid_z0, levels, colors=['black'])
  #cnt = ax.contour(v_x,v_y,(Z_core/np.max(Z_core)), levels, colors=['black']) #try2
  #plt.contour(v_xa,v_ya, grid_z0, levels, colors=['black'])
  
  ###### Pcolor approach
  cnt = ax.pcolor(v_xa,v_ya,( grid_z0), cmap='viridis',shading='auto') # Alfv norm
  #cnt = ax.pcolor(v_x,v_y,np.log10( (Z_core+Z_beam)/np.max((Z_core+Z_beam))    ), cmap='viridis') # Thermal speed norm
  
  ax.axis('off')
  ax.set_aspect('equal', 'box')
  #cnt.set_clim(vmin=-1, vmax=0) # for log-scale normalization
  cnt.set_clim(vmin=0.01, vmax=1) # for lin-scale normalization
  #plt.show()
  
  filename = r'C:\Users\vechd\.spyder-py3\instability_calc\VDF_images\\' + str(i).zfill(5) + '_QQ.jpg'
  fig.savefig(filename,bbox_inches='tight')
   
  plt.clf()
  fig.clear('all')
  plt.close(fig)
  plt.close()
  ### resize image and save
  image = Image.open(filename)
  new_image = (image.resize((100, 100)))
  new_image.save(filename)
  image.close()
  gc.collect()
  print(str(i))
  
  
  
### testing create_image.py  
from create_image import create_image
for i in range(0,len(n_core)):
  create_image(dim_params,i, v_x,v_y, v_xa, v_ya)    