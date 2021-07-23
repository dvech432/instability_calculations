#### a function that takes 9 inputs and plots a 3-D VDF


def get_fv(u,w, dV,q,m,v,n):
 import math

 D=(q*dV/(2*m*w**2))
 w1 = (w/v)
 u1 = (u/v)
 phi=[]
 for i in range(0,len(v)):     
   phi.append(( (1*n)/(w/1000))*math.exp(-0.5*(((v[i]-u)/w)**2)))

    
 T1=phi*dV
 T2= (1+ ((1/8)*(1-(u1*(2+w1**2))+u1**2)*D**2))
 T3 = (  (1/192)*( 1- (u1*(4+ (6*w1**2) + (12*w1**4) + (15*w1**6))) + (3*u1**2*(2 + (4*w1**2) + (5*w1**4) )) - (u1**3*(4 + (6*w1**2))) + u1**4 )*D**4)
 f_core= T1*(T2+T3)

 return f_core