#### function that generates and saves VDFs images that include up to 3 particle populations

def create_image_helios_3comps(dim_params, v_x,v_y, v_xa, v_ya):
  import numpy as np
  import pandas as pd
  
  m=1.67272*10**-27 # proton mass
  q=1.60217662*10**-19 # charge
  kb=1.38064852*10**-23 # boltzmann constant
  c=299792458 # speed of light
  #### VDF params

  v_core=dim_params[:,2]
  v_beam=dim_params[:,6]
  v_alpha=dim_params[:,10]

  n_core=dim_params[:,1]
  n_beam=dim_params[:,5]
  n_alpha=dim_params[:,9]

  T_core_perp=dim_params[:,4]
  T_core_par=dim_params[:,3]

  vth_core_perp=(2*kb*T_core_perp/m)**(0.5)
  vth_core_par=(2*kb*T_core_par/m)**(0.5)

  T_beam_perp=dim_params[:,8]
  T_beam_par=dim_params[:,7]

  vth_beam_perp=(2*kb*T_beam_perp/m)**(0.5)
  vth_beam_par=(2*kb*T_beam_par/m)**(0.5)

  T_alpha_perp=dim_params[:,12]
  T_alpha_par=dim_params[:,11]

  vth_alpha_perp=(2*kb*T_beam_perp/(4*m))**(0.5)
  vth_alpha_par=(2*kb*T_beam_par/(4*m))**(0.5)

  ###### B-field
  b0 = dim_params[:,0]
  n_total=np.squeeze(np.nansum(np.dstack((n_core,n_beam,n_alpha)),2))
  alfv = (b0*10**-9./np.sqrt(4*np.pi*10**-7.*(n_total)*100**3*1.67*10**-27))

  import matplotlib.pyplot as plt
  from scipy.interpolate import griddata
  import gc

  from PIL import Image
  
  #for i in range(0,len(n_core)):
  for i in range(26000,len(n_core)):    
    Z_core=n_core[i]*(m/(2*np.pi*kb*T_core_perp[i]))*(m/(2*np.pi*kb*T_core_par[i]))**(0.5)*np.exp(-m*((v_x)**2)/(2*kb*T_core_par[i]))*np.exp(-m*((v_y)**2)/(2*kb*T_core_perp[i]))  
    Z_beam=n_beam[i]*(m/(2*np.pi*kb*T_beam_perp[i]))*(m/(2*np.pi*kb*T_beam_par[i]))**(0.5)*np.exp(-m*(((v_beam[i]-v_core[i])-v_x)**2)/(2*kb*T_beam_par[i]))*np.exp(-m*((v_y)**2)/(2*kb*T_beam_perp[i]))
    Z_alpha=n_alpha[i]*((4*m)/(2*np.pi*kb*T_alpha_perp[i]))*((4*m)/(2*np.pi*kb*T_alpha_par[i]))**(0.5)*np.exp(-(4*m)*(((v_alpha[i]-v_core[i])-v_x)**2)/(2*kb*T_alpha_par[i]))*np.exp(-(4*m)*((v_y)**2)/(2*kb*T_alpha_perp[i]))
    
    Z_core[np.isnan(Z_core)] = 0
    Z_beam[np.isnan(Z_beam)] = 0
    Z_alpha[np.isnan(Z_alpha)] = 0
 
    
    f_total=Z_core+Z_beam+Z_alpha
    ### adding noise to the total VDF
    r1 = np.reshape( np.random.uniform(0,2*np.pi,  len(Z_core.flatten()) ), np.shape(Z_core) )
    r2 = np.reshape( np.random.uniform(0,2*np.pi, len(Z_core.flatten()) ), np.shape(Z_core) )

    #s_noisy= np.sqrt( (f_total + (0.0000001*np.sin(r1)) )**2 + (0.0000001*np.sin(r2))**2) #+(0.00001*np.reshape(np.random.normal(0,1,len(Z_core.flatten())),np.shape(Z_core))*f_total ) 
    
    #### original params that were used to generated 3comps_log and 3comps_lin
    s_noisy = f_total + (0.08*np.reshape(np.random.normal(0,1,len(Z_core.flatten())),np.shape(Z_core))*f_total ) + \
    np.abs(0.0000000000000001*np.sin(r1))
    #### new params that are used to generated 3_comps_log_noisy
    #s_noisy = f_total + (0.25*np.reshape(np.random.normal(0,1,len(Z_core.flatten())),np.shape(Z_core))*f_total ) + \
    #np.abs(0.0000000000000001*np.sin(r1))
    
    ### interpolation to the Alfv normalized grid
    v_xn= v_x/alfv[i]
    v_yn=v_y/alfv[i]
    points = np.array( (v_xn.flatten(), v_yn.flatten()) ).T
    values = s_noisy.flatten()/np.max(s_noisy.flatten()) # noisy
    #values = ((Z_core.flatten()+Z_beam.flatten())/np.max(Z_core.flatten()+Z_beam.flatten())) # noise free
    
    #grid_z0 = griddata(points, values, (v_xa, v_ya), method='linear')
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
    ##### NOTE: np.log10 is on
    cnt = ax.pcolor(v_xa,v_ya, np.log10( grid_z0), cmap='viridis',shading='auto') # Alfv norm
    #cnt = ax.pcolor(v_x,v_y,np.log10( (Z_core+Z_beam)/np.max((Z_core+Z_beam))    ), cmap='viridis') # Thermal speed norm
  
    ax.axis('off')
    ax.set_aspect('equal', 'box')
    cnt.set_clim(vmin=-1, vmax=0) # for log-scale normalization
    #cnt.set_clim(vmin=0.01, vmax=1) # for lin-scale normalization
  
    filename = r'D:\Research\Data\DISP\VDF_images_helios_3comps_log\\' + str(i).zfill(5) + '_QQ.jpg'
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