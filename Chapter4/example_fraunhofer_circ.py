from GeneralFunc import *
import matplotlib.pyplot as plt
import numpy as np

N = 512                     #% number of grid points per side 
L = 7.5e-3                  #% total size of the grid[m] 
d1 = L / N                  #% source-plane grid spacing[m] 
D = 1e-3                    #% diameter of the aperture[m] 
wvl = 1e-6                  #% optical wavelength[m] 
k = 2 * np.pi / wvl
Dz = 20                     #% propagation distance[m]

x1, y1 = np.meshgrid(np.arange(-N/2, N/2-1)*d1, np.arange(-N/2, N/2-1)*d1)
Uin = circ1(x1, y1, D)
Uout, x2, y2 = fraunhofer_prop(Uin, wvl, d1, Dz)

# Uout_th = np.exp(1j*k/(2*Dz)*(x2**2+y2**2))/(1j*wvl*Dz)*D**2*np.pi/4
#         *jinc(D*np.sqrt(x2**2+y2**2)/(wvl * Dz))

plt.figure()
plt.imshow(np.real(Uout), cmap='gray')
plt.show()