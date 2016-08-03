'''physics'''
import numpy as np;
e0 = 8.85418782e-12;
c = 2.99792458e8;
e = 1.60217657e-19
G=6.67408e-11
h=6.626070040e-34
e=1.602176208e-19
hc=h*c*1e9/e;
alpha = e**2/(4*np.pi*e0)/(h*c/(2*np.pi));
m_e=9.10938356e-31
m_e_cgs= m_e *1e3
mu0 = 4*np.pi*1e-7 
r_e = e**2/m_e/c**2/(4*np.pi*e0)
kb = 8.6173324e-5
kb_si = 1.38064852e-23
a0 = lambda I,l=.8e-4: np.sqrt(r_e/c/m_e/(c**2)*2/np.pi * I * l**2)

debye = lambda T_eV, ne: np.sqrt(T_eV/(hc*1e-7)/(2*alpha)/ne)
nc = lambda l,gm=1,m=m_e,q=e: e0*m*(2*np.pi*c/l)**2/q**2/gm*1e-6
wp = lambda ne,q=e,m=m_e: np.sqrt(ne*e**2/m/e0)
ItoE = lambda I: np.sqrt(2*I*1e4/(e0*c));
EtoI = lambda E: e0*c*E**2/2.0*1e-4


def laserE(I=None,E_0=None,
           T=30e-15, w=2.26e-6,dim="3D"):
    '''
    Get total energy in a Gaussian Laser in Joules.
    

    Parameters and Keywords
    -----------------------
    I     -- laser intensity in W/cm^2
    E_0   -- Peak E field in V/m
    T     -- FWHM of the pulse in seconds.
    w     -- Spotsize in meters.
    dim   -- Spatial dimension, either "2D", or "3D" or None for "3D"

    Returns laser energy.
    '''
    if not E_0 and not I:
        raise ValueError("Specify either the intensity or the peak E field");
    if I:
        E_0 = ItoE(I);
    if dim == "2D":
        return w * np.sqrt(np.pi/2) * (c*e0*E_0**2)/2 * T*1e-2;
    elif not dim or dim == "3D":
        return w**2 * (np.pi/2) * (c*e0*E_0**2)/2 * T;
    else:
        raise ValueError("dim is not None, '2D' or '3D'");

zr = lambda lm,w0: w0**2*np.pi/lm
waist = lambda z,lm,w0: w0*np.sqrt(1+(z/zr(lm,w0))**2)

def maxwellE(E,kT):
    return 2*np.sqrt(E/np.pi) * (1.0/kT)**1.5 * np.exp(-E/kT)
