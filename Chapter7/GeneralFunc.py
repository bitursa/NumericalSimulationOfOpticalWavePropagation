# -*- coding: utf-8 -*-
import numpy as np
from scipy.special import jn
import scipy.io as sio
from scipy.special import gamma
import scipy.integrate as integrate
from scipy.special import fresnel


def circ(r):
    return np.where(abs(r) <= 1, 1, 0)


def circ1(x, y, D):
    r = np.sqrt(x**2+y**2)
    return np.where(abs(r) <= D/2, 1, 0)


def rect(x):
    return np.where(abs(x) <= 0.5, 1, 0)


def tri(x):
    return np.where(abs(x) <= 1, 1-abs(x), 0)


def jinc(x):
    y = np.ones_like(x)
    i = 1
    while i <= len(x):
        y[i] = 2*jn(0, np.pi*x[i])/(np.pi*x[i])
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
    N = np.size(G, 1)
    g = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(G))) * (N*delta_f)**2
    return g


def myconv(A, B, delta):
    # one dimension
    N = len(A)
    res = ift(ft(A, delta)*ft(B, delta), 1/(N*delta))
    return res


def myconv2(A, B, delta):
    N = np.size(A, 1)
    res = ift2(ft2(A, delta)*ft2(B, delta), 1/(N*delta))
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
    idx = maskcorr != 0
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
    N = np.size(g, 0)
    F = 1/(N*delta)
    f_X = np.arange(-N/2, N/2-1)*F

    der = ift((1j*2*np.pi*f_X)**n*ft(g, delta), F)
    return der


def fraunhofer_prop(Uin, wvl, d1, Dz):
    N = np.size(Uin, 1)
    k = 2 * np.pi / wvl
    fx = np.arange(-N/2, N/2-1)/(N*d1)

    x2, y2 = np.meshgrid(wvl*Dz*fx, wvl*Dz*fx)

    # clear('fX')
    Uout = np.exp(1j*k/(2*Dz))/(1j * wvl * Dz) * ft2(Uin, d1)
    return Uout, x2, y2


def fresnel_prop_square_ap(x2, y2, D1, wvl, Dz):
    N_F = (D1/2)**2/(wvl*Dz)
    bigX = x2/np.sqrt(wvl*Dz)
    bigY = y2/np.sqrt(wvl*Dz)
    alpha1 = -np.sqrt(2)*(np.sqrt(N_F)+bigX)
    alpha2 = np.sqrt(2)*(np.sqrt(N_F)-bigX)
    beta1 = -np.sqrt(2)*(np.sqrt(N_F)+bigY)
    beta2 = np.sqrt(2)*(np.sqrt(N_F)-bigY)

    sa1, ca1 = fresnel(alpha1)
    sa2, ca2 = fresnel(alpha2)
    sb1, cb1 = fresnel(beta1)
    sb2, cb2 = fresnel(beta2)

    U = 1/(2j)*((ca2-ca1)+1j*(sa2-sa1))*((cb2-cb1)+1j*(sb2-sb1))
    return U


def lens_against_ft(Uin, wvl, d1, f):
    N = np.size(Uin, 1)
    k = 2*np.pi/wvl
    fX = np.arange(-N/2, N/2-1)/(N*d1)

    x2, y2 = np.meshgrid(wvl*f*fX, wvl*f*fX)

    Uout = np.exp(1j*k/(2*f)*(x2**2+y2**2))/(1j*wvl*f)*ft2(Uin, d1)
    return Uout, x2, y2


def lens_in_front_ft(Uin, wvl, d1, f, d):
    N = np.size(Uin, 1)
    k = 2*np.pi/wvl
    fX = np.arange(-N/2, N/2-1)/(N*d1)

    x2, y2 = np.meshgrid(wvl*f*fX, wvl*f*fX)
    Uout = 1/(1j*wvl*f)*np.exp(1j*k/(2*f)*(1-d/f) *
                               (x2**2+y2**2))/(1j*wvl*f)*ft2(Uin, d1)
    return Uout, x2, y2


def lens_behind_ft(Uin, wvl, d1, f):
    N = np.size(Uin, 1)
    k = 2*np.pi/wvl
    fX = np.arange(-N/2, N/2-1)/(N*d1)
    x2, y2 = np.meshgrid(wvl*f*fX, wvl*f*fX)

    Uout = f/d*1/(1j*wvl*f)*np.exp(1j*k/(2*f)*(x2**2+y2**2))*ft2(Uin, d1)
    return Uout, x2, y2


def zernike(i, r, theta):
    data = sio.loadmat(
        '/Users/ursa/Code/MyScript/NumericalSimulationOfOpticalWavePropagation/Chapter5/zernike_index.mat')
    n = data['zernike_index'][i-1, 0]
    m = data['zernike_index'][i-1, 1]
    if m == 0:
        Z = np.sqrt(n+1)*zrf(n, 0, r)
    else:
        if np.mod(i, 2) == 0:
            Z = np.sqrt(2*(n+1))*zrf(n, m, r)*np.cos(m*theta)
        else:
            Z = np.sqrt(2*(n+1))*zrf(n, m, r)*np.sin(m*theta)
    return Z


def zrf(n, m, r):
    R = 0
    for s in np.linspace(0, (n-m)/2, int((n-m)/2+1), endpoint=True):
        num = (-1)**s * gamma(n-s+1)
        denom = gamma(s+1)*gamma((n+m)/2-s+1)*gamma((n-m)/2-s+1)
        R = R + num/denom*(r**(n-2*s))
    return R


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)


def one_step_prop(Uin, wvl, d1, Dz):
    N = np.size(Uin, 1)
    k = 2*np.pi/wvl

    x1, y1 = np.meshgrid(np.arange(-N/2, N/2)*d1, np.arange(-N/2, N/2)*d1)
    x2, y2 = np.meshgrid(np.arange(-N/2, N/2)/(N*d1)*wvl *
                         Dz, np.arange(-N/2, N/2)/(N*d1)*wvl*Dz)
    Uout = 1/(1j*wvl*Dz)*np.exp(1j*k/(2*Dz)*(x2**2+y2**2)) * \
        ft2(Uin*np.exp(1j*k/(2*Dz))*(x1**2+y1**2), d1)
    return x2, y2, Uout


def two_step_prop(Uin, wvl, d1, d2, Dz):
    N = np.size(Uin, 1)
    k = 2*np.pi/wvl

    x1, y1 = np.meshgrid(np.arange(-N/2, N/2)*d1, np.arange(-N/2, N/2)*d1)
    m = d1/d2
    Dz1 = Dz/(1-m)
    d1a = wvl*np.abs(Dz1)/(N*d1)
    x1a, y1a = np.meshgrid(np.arange(-N/2, N/2)*d1a, np.arange(-N/2, N/2)*d1a)
    Uitm = 1/(1j*wvl*Dz1)*np.exp(1j*k/(2*Dz1))*(x1a**2+y1a**2) * \
        ft2(Uin*np.exp(1j*k/(2*Dz1))*(x1**2+y1**2), d1)
    Dz2 = Dz - Dz1
    x2, y2 = np.meshgrid(np.arange(-N/2, N/2)*d2, np.arange(-N/2, N/2)*d2)
    Uout = 1/(1j*wvl*Dz2)*np.exp(1j*k/(2*Dz2))*(x2**2+y2**2) * \
        ft2(Uitm*np.exp(1j*k/(2*Dz2))*(x1a**2+y1a**2), d1a)
    return x2, y2, Uout


def ang_spec_prop(Uin, wvl, d1, d2, Dz):
    N = np.size(Uin, 1)
    k = 2*np.pi/wvl

    x1, y1 = np.meshgrid(np.arange(-N/2, N/2)*d1, np.arange(-N/2, N/2)*d1)
    r1sq = x1**2+y1**2
    df1 = 1/(N*d1)
    fX, fY = np.meshgrid(np.arange(-N/2, N/2)*df1, np.arange(-N/2, N/2)*df1)
    fsq = fX**2+fY**2
    m = d2/d1
    x2, y2 = np.meshgrid(np.arange(-N/2, N/2)*d2, np.arange(-N/2, N/2)*d2)
    r2sq = x2**2+y2**2
    Q1 = np.exp(1j*k/2*(1-m)/Dz*r1sq)
    Q2 = np.exp(-1j*np.pi**2*2*Dz/m/k*fsq)
    Q3 = np.exp(1j*k/2*(m-1)/(m*Dz)*r2sq)
    Uout = Q3*ift2(Q2*(ft2(Q1*Uin/m, d1)), df1)
    return x2, y2, Uout


# def ft_phase_screen(r0, N, delta, L0, l0):
#     del_f = 1/(N*delta)
#     f1 = np.arange(-N/2, N/2-1)*del_f
#     fx, fy = np.meshgrid(f1,f1)
#     r, theta = cart2pol(fx,fy)
#     fm = 5.92/10/(2*np.pi)
#     f0 = 1/L0

#     PSD_phi = 0.023*r0**(-5/3)*np.exp(-(f/fm)**2)/(f**2+f0**2)*(11/6)
