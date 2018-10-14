# -*- coding: utf-8 -*-
import numpy as np
from scipy.special import jn
import scipy.io as sio
from scipy.special import gamma

def zrf(n,m,r):
    R = 0
    for s in np.linspace(0, (n-m)/2, int((n-m)/2+1), endpoint=True):
        num = (-1)**s * gamma(n-s+1)
        denom = gamma(s+1)*gamma((n+m)/2-s+1)*gamma((n-m)/2-s+1)
        R = R +num/denom*(r**(n-2*s))
    return R


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

N = 32
L = 2
delta = L/N

x, y = np.meshgrid(np.arange(-N/2, N/2)*delta, np.arange(-N/2, N/2)*delta)
r, theta = cart2pol(x, y)

# zernike(2, r, theta)
data = sio.loadmat(
    '/Users/ursa/Code/MyScript/NumericalSimulationOfOpticalWavePropagation/Chapter5/zernike_index.mat')
n = data['zernike_index'][2-1, 0]
m = data['zernike_index'][2-1, 1]

if m == 0:
    Z = np.sqrt(n+1)*zrf(n, 0, r)
else:
    if np.mod(2, 2) == 0:
        Z = np.sqrt(2*(n+1))*zrf(n, m, r)*np.cos(m*theta)
    else:
        Z = np.sqrt(2*(n+1))*zrf(n, m, r)*np.sin(m*theta)
