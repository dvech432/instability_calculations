## plotting VDFs in polar system



    # POLAR STUFF  
import numpy as np
rho = np.sqrt(v_xa.flatten()**2 + v_ya.flatten()**2)
phi = np.arctan2(v_ya.flatten(), v_xa.flatten())
    
rbins = np.linspace(0, 2.5,70)
abins = np.linspace(-np.pi, np.pi, 16)

hist, _, _ = np.histogram2d(phi, rho, bins=(abins, rbins))
A, R = np.meshgrid(abins, rbins)
    
plt.subplot2grid((2,2), (1,1), polar=True)
plt.scatter(A,R)

plt.pcolormesh(A, R, hist.T, cmap="viridis") #vmin=-6, vmax=-2
    # axs[1,1].grid(True)
    
    
    
from physt import histogram, binnings, special
import numpy as np
import matplotlib.pyplot as plt

# Generate some points in the Cartesian coordinates
np.random.seed(42)

x = v_xa.flatten()
y = v_xy.flatten()
z = np.log10( grid_z0)

# Create a polar histogram with default parameters
hist = special.polar_histogram(x, y)
ax = hist.plot.polar_map()

import random
import numpy as np
import matplotlib.pyplot as plt
from physt import special

# Generate some points in the Cartesian coordinates
np.random.seed(42)

gen = lambda l, h, s = 3000: np.asarray([random.random() * (h - l) + l for _ in range(s)])

X = gen(-100, 100)
Y = gen(-1000, 1000)
Z = gen(0, 1400)

hist = special.polar_histogram(X, Y, weights=Z, radial_bins=40)
# ax = hist.plot.polar_map()

hist.plot.polar_map(density=True, show_zero=False, cmap="inferno", lw=0.5, figsize=(5, 5))
plt.show()
    
    
    ##
    
    
import numpy as np; np.random.seed(42)
import matplotlib.pyplot as plt

# two input arrays
azimut = np.random.rand(3000)*2*np.pi
radius = np.random.rayleigh(29, size=3000)

# define binning
rbins = np.linspace(0,radius.max(), 30)
abins = np.linspace(0,2*np.pi, 60)

#calculate histogram
hist, _, _ = np.histogram2d(azimut, radius, bins=(abins, rbins))
A, R = np.meshgrid(abins, rbins)

# plot
fig, ax = plt.subplots(subplot_kw=dict(projection="polar"))

pc = ax.pcolormesh(A, R, hist.T, cmap="magma_r")
fig.colorbar(pc)

plt.show()