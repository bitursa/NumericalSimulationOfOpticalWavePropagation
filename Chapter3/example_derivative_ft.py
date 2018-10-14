from GeneralFunc import *
import matplotlib.pyplot as plt
import numpy as np

N = 64 
L = 6

delta = L/N
x = np.arange(-N/2,N/2-1)*delta 
w = 3
window = rect(x/w)
g = x**5*window

gp_samp = np.real(derivative_ft(g, delta, 1))*window
gpp_samp = np.real(derivative_ft(g, delta, 2))*window

gp = 5*x**4*window
gpp = 20*x**3*window