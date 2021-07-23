
import numpy as np
from rotate_v2 import rotate_v2

def get_coords(v_core,v,i,f_core):
       
  y0=np.zeros(np.reshape(v,(-1,1)).shape)
  x0=np.reshape( (v/1000), (-1,1))
  origin=(v_core[i],0)
  point= np.concatenate((x0,y0),axis=1)
  angle=np.linspace(0,180,9)
  qqx=[]
  qqy=[]
  qqz=[]
  for i in range(0,len(angle)):
    qx, qy = rotate_v2(point, origin=(v_core[i],0), degrees=angle[i])
    qqx.append(np.array(qx))
    qqy.append(np.array(qy))
    qqz.append(f_core)
    
  qqx=np.array(qqx)
  qqy=np.array(qqy)
  qqz=np.array(qqz)  
  return qqx, qqy, qqz