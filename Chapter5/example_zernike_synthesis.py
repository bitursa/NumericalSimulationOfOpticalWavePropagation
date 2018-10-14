from GeneralFunc import *
import numpy as np
import matplotlib.pyplot as plt

N = 40
L = 2
delta = L/N

x, y = np.meshgrid(np.arange(-N/2, N/2)*delta, np.arange(-N/2, N/2)*delta)
r, theta = cart2pol(x, y)

ap = circ1(x, y, 2)
idxAp = ap!=0

r0 = L/20
screen = ft_phase_screen(r0, N, delta, inf, 0)/(2*np.pi)*ap

W = screen*idxAp

nModes = 100
Z = np.zeros(np.size(W, nModes))

for i in np.linspace(1,nModes,nModes,endpoint=True):
    temp = zernike(i, r, theta)
    Z[:,i-1] = temp[idxAp]

A = np.linalg.lstsq(Z, W)
W_prime = Z*A

scr = np.zeros((N,N))
scr(idxAp) = W_prime