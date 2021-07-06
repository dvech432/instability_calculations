


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

V_high=np.logspace(np.log10(307.317073),np.log10(5121.95121),59);
V_low=np.logspace(np.log10(292.68292),np.log10(4878.04878),59);

dV=(V_high-V_low)

m = 1.67272*10**-27 # proton mass
q=1.60217662*10**-19 # charge

#G = normrnd(0,1,[1000000 1]); # add random noise to the VDF

from get_fv import get_fv
i=0
f_core = get_fv(v_core[i],vth_core[i], dV,q,m,v,n_core[i])    

