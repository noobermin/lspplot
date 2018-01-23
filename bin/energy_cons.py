#!/usr/bin/env python
'''
Usage:
  ./energy_cons.py [options] [<output>]

Options:
  --help -h
  --lsp=FILE              Supply the .lsp file.
  --intensity=I -I I      Specify laser intensity directly in W/cm^2
  --pulse=T -T T          Specify laser pulse FWHM directly in seconds.
  --w=W -w W              Specify laser spotsize directly in meters.
'''
import numpy as np;
from docopt import docopt;
from os import listdir;
import re;
opts=docopt(__doc__,help=True);
fs=listdir();
lspf = [f for f in fs
       if re.match(".*.lsp$",f)][-1]
hstf = [f for f in fs
       if re.match(".*history.p4$",f)][-1];
d = np.loadtxt(hstf).T;
#getting column of net energy probe
with open(hstf,'r') as f:
    for line in f.readlines():
        m = re.match("^#([0-9]+): *net energy",line);
        if m:
            column = int(m.group(1))+1;
            break;
    else:
        raise ValueError(
            "file {} does not have a net energy probe".format(
                hstf));

time = d[1];
E    = d[column];

e0 = 8.85418782e-12;
c = 2.99792458e8;
e = 1.60217657e-19
def laserE(E_0, T, w,dim="3D"):
    '''
    Get total energy in a Gaussian Laser.
    

    Parameters and Keywords
    -----------------------
    E_0   -- Peak E field.
    T     -- FWHM of the pulse.
    w     -- Spotsize.
    dim   -- Spatial dimension, either "2D"q, or "3D" or None for "3D"

    Returns laser energy.
    '''

    if dim == "2D":
        return w * np.sqrt(np.pi/2) * (c*e0*E_0**2)/2 * T*1e-2;
    elif not dim or dim == "3D":
        return w**2 * (np.pi/2) * (c*e0*E_0**2)/2 * T;
    else:
        raise ValueError("dim is not None, '2D' or '3D'");
if not (opts['--intensity'] and opts['--pulse'] and opts['--w']):
    raise NotImplementedError("Non explicit quantities not implemented yet.")
I = float(opts['--intensity']);
E_0 = np.sqrt(2*I*1e4/(c*e0));
T = float(opts['--pulse']);
w = float(opts['--w']);

e_laser = laserE(E_0,T,w,dim='2D');
import matplotlib
if opts['<output>']:
    matplotlib.use("agg");
import matplotlib.pyplot as plt;
plt.plot(time*1e6, E/e_laser);
plt.xlabel("time (fs)");
plt.ylabel("energy ratio");
if opts['<output>']:
    plt.savefig(opts['<output>']);
else:
    plt.show();
