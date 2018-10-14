# -*- coding: utf-8 -*-
import numpy as np


def circ(r):
    return np.where(abs(r) <= 1, 1, 0)

def rect(x):
    return np.where(abs(x) <= 0.5, 1, 0)

def ft(g, delta):
    G = np.fft.fftshift(np.fft.fft(np.fft.fftshift(g))) * delta
    return G 

def ift(G, delta_f):
    g = np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(G))) * delta_f
    return g
