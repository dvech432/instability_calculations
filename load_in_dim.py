### load in dim files, format and save them
import pandas as pd
import numpy as np
import os
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
dim_params=pd.DataFrame(dim_params)
dim_params.to_csv(filename,index=False)

