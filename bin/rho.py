#!/usr/bin/env python2
# -*- coding: utf-8 -*-
'''
Render charge density by divergence of E. Only works for uniform spacing right now.

Usage:
    ./sclrq.py [options] (--show|-s) <i>
    ./sclrq.py [options] <i> <outname>

Options:
    --help -h
    --show -s           Show
    --nozip -U          flds are NOT gzipped.
    --log10 -l          Log it.
    --lims=LIM          Set lims [default: (1e18,1e23)]
    --highlight=H       Set highlight.
    --quantity=Q -Q Q   Use this quantity for n_c lineout. [default: RhoN10]
    --dir=D -D D        Read from this dir [default: .]
    --restrict=R        Restrict it.
    --title=T           Set the title [default: Electron density]
    --units=U           Set the colorbar units [default: e/cc ]
    --laser             Plot contours of the laser poyting vector.
    --intensity=I -I I  Make a contour of this intensity [default: 3e18]
    --linthresh=L       Set the linear threshold for SymLogPlot [default: 1.0]
    --linscale=L        Set the linear threshold for SymLogPlot [default: 1.0]
    --cmap=CMAP         Set the colormap. [default: viridis]

'''
from docopt import docopt;
import numpy as np;
import numpy.linalg as lin;
from pys import parse_ftuple, parse_ituple;
from lspreader.flds import read_indexed, restrict
from lspplot.sclr import S;
from lspplot.pc import pc,highlight;
from lspplot.consts import c,mu0,e0;
e=1.602176208e-19
opts = docopt(__doc__,help=True);
gzip = not opts['--nozip'];
quantity = opts['--quantity'];

fvar=['E']
if opts['--laser']:
    fvar+=['B'];
titlestr=opts['--title']
units=opts['--units'];
svar=[quantity];
#####################################
#reading data
d = read_indexed(int(opts['<i>']),
    flds=fvar,sclr=svar,
    gzip=gzip,dir=opts['--dir'],
              gettime=True,vector_norms=False);
if opts['--restrict']:
    res = parse_ituple(opts['--restrict'],length=None);
    restrict(d,res);

#massaging data
t  = d['t'];
x,y = d['x']*1e4,d['y']*1e4
Ex,Ey = d['Ex']*1e5, d['Ey']*1e5;
if np.isclose(y.max(),y.min()):
    y = d['z']*1e4
    Ey = d['Ez']*1e5;
q = d[quantity];
rho = reduce(
    np.add,
    np.gradient(
        [Ex,Ey],
        varargs=[x[:,0], y[0,:]])
) / e0 / e;

#####################################
#plotting

#getting options from user
mn,mx = parse_ftuple(opts['--lims'],length=2);


#plot the density
title="{}\nTime: {:.2f} fs".format(titlestr,t*1e6);
r=pc(
    rho,(x,y), lims=(mn,mx),log=opts['--log10'],
    clabel=units, title=title,
    agg=not opts['--show']);

if opts['--highlight']:
    myhi  = float(opts['--highlight']);
    highlight(
        r, myhi,
        color="lightyellow", alpha=0.5);

if opts['--laser']:
    laser = S(d);
    I = float(opts['--intensity']);
    highlight(r, I, q=laser,
              color="red", alpha=0.15);
    
import matplotlib.pyplot as plt;
if opts['--show']:
    plt.show();
else:
    plt.savefig(opts['<outname>']);


