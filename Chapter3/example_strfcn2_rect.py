from GeneralFunc import *
import matplotlib.pyplot as plt
import numpy as np

N = 256         # % number of samples
L = 16           # % grid size[m]
delta = L / N   # % sample spacing[m]
F = 1/L         # % frequency-domain grid spacing[1/m]
x = np.arange(-N/2, N/2-1, 1) * delta
y = np.arange(-N/2, N/2-1, 1) * delta
X, Y = np.meshgrid(x, x)
w = 2  # % width of rectangle
A = rect(X/w)*rect(Y/w)
mask = np.ones([N-1,N-1])

C = str_fcn2_ft(A, mask, delta)/delta**2

C_cont = 2*w**2*(1-tri(X/w)*tri(Y/w))

# print(C.shape,C_cont.shape)
plt.figure(1)
plt.imshow(np.real(C), cmap='gray')
plt.figure(2)
plt.imshow(C_cont, cmap='gray')
plt.tight_layout()
plt.show()
