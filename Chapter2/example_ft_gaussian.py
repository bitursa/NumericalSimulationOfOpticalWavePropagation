# -*- coding: utf-8 -*-
from GeneralFunc import *
import numpy as np
import matplotlib.pyplot as plt

# function values to be used in DFT 
L = 5                   # spatial extent of the grid 
N = 32                  #% number of samples 
delta = L / N           # sample spacing 
x = np.arange(-N/2, N/2-1, 1) * delta
f = np.arange(-N/2, N/2-1, 1) / (N * delta)
a = 1                   # sampled function & its DFT 
g_samp = np.exp(-np.pi*a*x**2)                  #% function samples 
g_dft = ft(g_samp, delta)                       #% DFT % analytic function & its continuous FT 
M = 1024

x_cont = np.linspace(x[0], x[-1], M)
f_cont = np.linspace(f[0], f[-1], M)
g_cont = np.exp(-np.pi*a*x_cont**2)
g_ft_cont = np.exp(-np.pi*f_cont**2/a)/a

plt.subplot(131)
plt.plot(x,g_samp,'*-')
plt.grid(True, linestyle=':')
plt.subplot(132)
plt.plot(f,np.real(g_dft),'*-')
plt.grid(True, linestyle=':')
plt.subplot(133)
plt.plot(x, np.imag(g_dft),'*-')
plt.grid(True, linestyle=':')
axes = plt.gca()
axes.set_ylim([-1,1])
plt.tight_layout()
plt.show()
