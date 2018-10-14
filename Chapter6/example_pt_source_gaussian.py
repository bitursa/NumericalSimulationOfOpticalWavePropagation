from GeneralFunc import *
import numpy as np
import matplotlib.pyplot as plt
D = 8e-3
wvl = 1e-6
k = 2 * np.pi / wvl
Dz = 1
arg = D/(wvl * Dz)
delta1 = 1/(10 * arg)
delta2 = D/100
N = 1024

x1, y1 = np.meshgrid(np.arange(-N/2, N/2)*delta1, np.arange(-N/2, N/2)*delta1)
r1, theta1 = cart2pol(x1, y1)

A = wvl*Dz
pt = A*np.exp(-1j*k/(2*Dz)*r1**2)*arg**2*np.sinc(arg*x1)*np.sinc(arg*y1)*np.exp(-(arg/4*r1)**2)

x2, y2, Uout = ang_spec_prop(pt, wvl, delta1, delta2, Dz)
