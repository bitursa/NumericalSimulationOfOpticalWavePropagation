from GeneralFunc import *
import matplotlib.pyplot as plt
import numpy as np

N = 64          # % number of samples 
L = 8           # % grid size[m] 
delta = L / N   # % sample spacing[m] 
F = 1/L         # % frequency-domain grid spacing[1/m] 
x = np.arange(-N/2,N/2-1,1) * delta
w = 2           #% width of rectangle 
A = rect(x/w)
B = A           # % signal 
C = myconv(A, B, delta)     #% perform digital convolution 
#% continuous convolution 
C_cont = w * tri(x/w)

plt.figure(1)
plt.subplot(311)
plt.plot(x,A,'*b')
plt.subplot(312)
plt.plot(x,B,'*r')
plt.subplot(313)
plt.plot(x,np.real(C),'*g')
plt.tight_layout()
plt.show()

