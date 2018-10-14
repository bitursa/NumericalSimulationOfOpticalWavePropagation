from GeneralFunc import *
import numpy as np
import matplotlib.pyplot as plt

N = 1024
L = 1e-2
delta1 = L/N
D = 2e-3
wvl = 1e-6
k = 2*np.pi/wvl
Dz = 1

x1, y1 = np.meshgrid(np.arange(-N/2, N/2)*delta1, np.arange(-N/2, N/2)*delta1)
ap = rect(x1/D)*rect(y1/D)
delta2 = wvl*Dz/(N*delta1)
x2, y2, Uout = two_step_prop(ap, wvl, delta1, delta2, Dz)

Uout_an = fresnel_prop_square_ap(x2[int(N/2)+1, :], 0, D, wvl, Dz)
