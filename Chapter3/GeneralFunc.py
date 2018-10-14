# -*- coding: utf-8 -*-
import numpy as np
from scipy.special import jn

def circ(r):
    return np.where(abs(r) <= 1, 1, 0)

def rect(x):
    return np.where(abs(x) <= 0.5, 1, 0)

def tri(x):
    return np.where(abs(x) <=1, 1-abs(x), 0)

def jinc(x):
    y = np.ones_like(x)
    i = 1
    while i <= len(x):
        y[i] = 2*jn(0,np.pi*x[i])/(np.pi*x[i])
        i = i+1
    return y

def ft(g, delta):
    G = np.fft.fftshift(np.fft.fft(np.fft.fftshift(g))) * delta
    return G 

def ft2(g, delta):
    G = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(g))) * delta**2
    return G

def ift(G, delta_f):
    g = np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(G))) * delta_f
    return g

def ift2(G, delta_f):
    N = np.size(G,1)
    g = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(G))) * (N*delta_f)**2
    return g

def myconv(A, B, delta):
    # one dimension
    N = len(A)
    res = ift(ft(A, delta)*ft(B, delta), 1/(N*delta))
    return res 

def myconv2(A, B, delta):
    N = np.size(A, 1)
    res = ift2(ft2(A,delta)*ft2(B,delta), 1/(N*delta))
    return res

def corr2_ft(u1, u2, mask, delta):
    N = np.size(u1, 1)
    c = np.zeros(N)
    delta_f = 1/(N*delta)

    U1 = ft2(u1*mask, delta)
    U2 = ft2(u2*mask, delta)
    U12corr = ift2(np.conj(U1)*U2, delta_f)

    maskcorr = ift2(np.abs(ft2(mask, delta))**2, delta_f)*delta**2
    ones = np.ones_like(maskcorr)
    idx = maskcorr!=0
    c[idx] = U12corr[idx]/maskcorr[idx]*mask[idx]
    return c


def str_fcn2_ft(ph, mask, delta):
    N = np.size(ph, 1)
    ph = ph*mask

    P = ft2(ph, delta)
    S = ft2(ph**2, delta)
    W = ft2(mask, delta)
    delta_f = 1/(N*delta)
    w2 = ift2(W*np.conj(W), delta_f)
    
    D = 2*ift2(np.real(S*np.conj(W))-np.abs(P)**2, delta_f)/w2*mask
    return D

def derivative_ft(g, delta, n):
    N = np.size(g,0)
    F = 1/(N*delta)
    f_X = np.arange(-N/2,N/2-1)*F

    der = ift((1j*2*np.pi*f_X)**n*ft(g, delta), F)
    return der 