#### generate VDF images from the Helios params

##### generating images of the VDFs, using dim_params.txt
import numpy as np
import pandas as pd

#### read in params
import os
### read in PLUME input/output
filename = r'C:\Users\vechd\.spyder-py3\instability_calc\helios_files\Helios_CB.h5'
helios_df = pd.read_hdf(filename)
##### pick the required params from the data frame
dim_params=np.array(helios_df[['B [nT]', 'n_p_core [cm^-3]','v_sw_core [m/s]','T_par_core [eV]','T_perp_core [eV]',
               'n_p_beam [cm^-3]','v_sw_beam [m/s]','T_par_beam [eV]','T_perp_beam [eV]']])

m=1.67272*10**-27 # proton mass
q=1.60217662*10**-19 # charge
kb=1.38064852*10**-23 # boltzmann constant
c=299792458 # speed of light

##### eV to Kelvin conversion
dim_params[:,4]=11604.525*dim_params[:,4]
dim_params[:,3]=11604.525*dim_params[:,3]
dim_params[:,8]=11604.525*dim_params[:,8]
dim_params[:,7]=11604.525*dim_params[:,7]

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
v_xa, v_ya= np.meshgrid(np.linspace(-2,2,100),np.linspace(-2,2,100))

import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import gc

from PIL import Image

### testing create_image.py  
from create_image import create_image
create_image(dim_params, v_x,v_y, v_xa, v_ya)    
