import pandas as pd
import numpy as np
import os
### read in PLUME input/output
filename = r'C:\Users\vechd\.spyder-py3\instability_calc\Vech_ML.h5'
plume = pd.read_hdf(filename)
### read in dimensional plasma params
folder_content=os.listdir(r'C:\Users\vechd\.spyder-py3\instability_calc\dim_files')
dim_params=[]
for i in range(1,len(folder_content)+1):
#for i in range(1,2):
  filename = r'C:\Users\vechd\.spyder-py3\instability_calc\dim_files\\' + 'vech_ML_' + str(i).zfill(7) + '.dim'
  p1 = np.genfromtxt(filename, dtype=float, invalid_raise=False, missing_values='', usemask=False, filling_values=0.0, skip_header=0)
  p2 = np.genfromtxt(filename, dtype=float, invalid_raise=False, missing_values='', usemask=False, filling_values=0.0, skip_header=1)
  p3=np.concatenate([np.reshape(p1,(-1,1)), np.reshape(p2[0:2,:],(-1,1))])
  dim_params.append(p3)
  print(str(i))
dim_params=np.squeeze(dim_params,axis=2)

### convert temperatures to kelvin

dim_params[:,3]=dim_params[:,3]*11604.525
dim_params[:,4]=dim_params[:,4]*11604.525
dim_params[:,7]=dim_params[:,7]*11604.525
dim_params[:,8]=dim_params[:,8]*11604.525

filename = r'C:\Users\vechd\.spyder-py3\instability_calc\dim_params.txt'
dim_params.to_csv(filename,index=False)


### Open and save Helios data

import pandas as pd
import numpy as np
import os
### read in PLUME input/output
filename = r'C:\Users\vechd\.spyder-py3\instability_calc\helios_files\Helios_CB.h5'
helios = pd.read_hdf(filename)



#### convertin back to physical units
# m=1.67272*10**-27 # proton mass
# q=1.60217662*10**-19 # charge
# kb=1.38064852*10**-23 # boltzmann constant
# c=299792458 # speed of light


# vth_core_par = df['$v_t/c$']*c
# T_core_par = (m*(vth_core_par**2))/(2*kb)
# T_core_perp = df['$\\alpha_c$']*T_core_par
# vth_core_perp= (2*kb*T_core_perp/m)**(0.5)


# T_beam_par = T_core_par/df['$\\tau_b$']
# vth_beam_par = (2*kb*T_beam_par/m)**(0.5)
# T_beam_perp = T_beam_par*df['$\\alpha_b$']
# vth_beam_perp = (2*kb*T_beam_perp/m)**(0.5)
