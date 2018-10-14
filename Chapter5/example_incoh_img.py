from GeneralFunc import *
import numpy as np
import matplotlib.pyplot as plt

N = 32  # number of grid points per side
L = 0.1  # total size of the grid [m]
D = 0.07  # diameter of pupil [m]
delta = L/N  # grid spacing [m]
wvl = 1e-6  # optical wavelength [m]
z = 0.25  # image distance [m]

x, y = np.meshgrid(np.arange(-N/2, N/2)*delta, np.arange(-N/2, N/2)*delta)
r,theta = cart2pol(x,y)

W = 0.05*zernike(4, 2*r/D, theta)
P = circ1(x, y, D)*np.exp(1j*2*np.pi*W)

h = ft2(P, delta)
delta_u = wvl*z/(N*delta)

u, v = np.meshgrid(np.arange(-N/2, N/2)*delta_u, np.arange(-N/2, N/2)*delta_u)
obj = (rect((u-1.4e-4)/5e-5)+rect(u/5e-5)+rect((u+1.4e-4)/5e-5))*rect(v/2e-4)

img = myconv2(np.abs(obj)**2, np.abs(h), 1)


# plt.figure()
# plt.imshow(obj)
plt.figure()
plt.imshow(np.abs(h))
# plt.figure()
# plt.imshow(np.real(img))
plt.show()
