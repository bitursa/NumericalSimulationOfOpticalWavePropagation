from GeneralFunc import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lstsq

N = 256
L = 2
delta = L/N

x, y = np.meshgrid(np.arange(-N/2, N/2)*delta, np.arange(-N/2, N/2)*delta)
r, theta = cart2pol(x, y)

ap = circ1(x, y, 2)

z2 = zernike(2, r, theta)*ap
z4 = zernike(4, r, theta)*ap
z21 = zernike(21, r, theta)*ap

W = 0.5*z2+0.25*z4-0.6*z21
W_image = W
idx = ap!=0
W = W[idx]
W1 = np.reshape(W, (np.size(W), 1))
Z = np.array((z2[idx],z4[idx],z21[idx]))
Z = np.transpose(Z)
A = np.linalg.lstsq(Z, W, rcond=None)
# print(A)

plt.figure()
plt.imshow(W_image)
# plt.figure()
# plt.imshow(z4)
# plt.figure()
# plt.imshow(z21)
plt.show()
