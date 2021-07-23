################## bi-maxwellian plots

##################### generating proton core, beam, alpha with random values
import numpy as np

m=1.67272*10**-27 # proton mass
q=1.60217662*10**-19 # charge
kb=1.38064852*10**-23 # boltzmann constant

S=10000

####################### core
a = 400*1000
b = 600*1000
v_core = np.random.uniform(a,b,S)

a = 20*1000
b = 80*1000
vth_core_par = np.random.uniform(a,b,S)
T_core_par = (m*(vth_core_par**2))/(2*kb)

a = 0.5
b = 1.5
vth_core_perp = vth_core_par*np.random.uniform(a,b,S)
T_core_perp = (m*vth_core_perp**2)/(2*kb)

a = 1
b = 100
n_core = np.random.uniform(a,b,S)

####################### beam

a = 10*1000
b = 50*1000
v_beam = v_core+ np.random.uniform(a,b,S)

a = 30*1000
b = 90*1000
vth_beam_par = np.random.uniform(a,b,S)
T_beam_par = (m*(vth_beam_par**2))/(2*kb)

a = 0.5
b = 1.5
vth_beam_perp = vth_beam_par*np.random.uniform(a,b,S)
T_beam_perp = (m*vth_beam_perp**2)/(2*kb)

a = 0.05
b = 1
n_beam = n_core*np.random.uniform(a,b,S)

######
a = 1
b = 10
b0 = np.random.uniform(a,b,S)
alfv = (b0*10**-9./np.sqrt(4*np.pi*10**-7.*(n_core+n_beam)*100**3*1.67*10**-27))

##########
###### boltzman eq

v_x, v_y=np.meshgrid(np.linspace(-200*1000,200*1000,100),np.linspace(-200*1000,200*1000,100))
 
v_xa, v_ya= np.meshgrid(np.linspace(-8,8,100),np.linspace(-8,8,100))

import matplotlib.pyplot as plt
from scipy.interpolate import griddata

for i in range(0,100):
  Z_core=n_core[i]*(m/(2*np.pi*kb*T_core_perp[i]))*(m/(2*np.pi*kb*T_core_perp[i]))**(0.5)*np.exp(-m*((v_x)**2)/(2*kb*T_core_perp[i]))*np.exp(-m*((v_y)**2)/(2*kb*T_core_par[i]))
  Z_beam=n_beam[i]*(m/(2*np.pi*kb*T_beam_perp[i]))*(m/(2*np.pi*kb*T_beam_perp[i]))**(0.5)*np.exp(-m*(((v_beam[i]-v_core[i])-v_x)**2)/(2*kb*T_beam_perp[i]))*np.exp(-m*((v_y)**2)/(2*kb*T_beam_par[i]))
  ### interpolation to the Alfv normalized grid
  v_xn= v_x/alfv[i]
  v_yn=v_y/alfv[i]
  points = np.array( (v_xn.flatten(), v_yn.flatten()) ).T
  values = ((Z_core.flatten()+Z_beam.flatten())/np.max(Z_core.flatten()+Z_beam.flatten()))
  
  grid_z0 = griddata(points, values, (v_xa, v_ya), method='linear')
  #grid_z0 = np.nan_to_num(grid_z0, nan=0)
  fig = plt.figure()
  ax = fig.add_axes([0, 0, 1, 1])
  levels=np.logspace(-1,0,10)
  cnt = ax.contour(v_xa,v_ya, grid_z0, levels, cmap='viridis')
  #ax.imshow(grid_z0)
  ax.axis('off')
  plt.show()
  filename = r'C:\Users\vechd\.spyder-py3\instability_calc\VDF_images\\' + str(i) + '_QQ.jpg'
  fig.savefig(filename)
  
  
  
  