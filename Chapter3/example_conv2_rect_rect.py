from GeneralFunc import *
import matplotlib.pyplot as plt
import numpy as np

N = 256         # % number of samples
L = 16           # % grid size[m]
delta = L / N   # % sample spacing[m]
F = 1/L         # % frequency-domain grid spacing[1/m]
x = np.arange(-N/2, N/2-1, 1) * delta
y = np.arange(-N/2, N/2-1, 1) * delta
X, Y = np.meshgrid(x,x)
w = 2  # % width of rectangle
A = rect(X/w)*rect(Y/w)
'''
B = rect(X/w)*rect(Y/w)           # % signal
C = myconv2(A, B, delta)  # % perform digital convolution
#% continuous convolution
C_cont = w**2 * tri(X/w) * tri(Y/w)
'''
mask = np.ones([N-1,N-1])
C = corr2_ft(A, A, mask, delta)
C_cont = w**2 * tri(X/w) * tri(Y/w)
plt.figure(1)
plt.imshow(np.real(C), cmap='gray', extent=[x[0], x[-1], y[0], y[-1]])
plt.figure(2)
plt.imshow(C_cont, cmap='gray', extent=[x[0], x[-1], y[0], y[-1]])
plt.tight_layout()
plt.show()
