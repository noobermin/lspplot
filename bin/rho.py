#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Render charge density by divergence of E. Only works for uniform spacing right now.

Usage:
    ./sclrq.py [options] (--show|-s) <i>
    ./sclrq.py [options] <i> <outname>

Options:
    --help -h
    --show -s           Show
    --nozip -U          sclr/flds are NOT gzipped.
    --zip   -Z          sclr/flds are gzipped. If neither of these two are set,
                        guess based on name.
    --log10 -l          Log it.
    --lims=LIM          Set lims [default: (1e18,1e23)]
    --highlight=H       Set highlight.
    --quantity=Q -Q Q   Use this quantity for n_c lineout. [default: RhoN10]
    --dir=D -D D        Read from this dir [default: .]
    --restrict=R        Restrict it.
    --title=T           Set the title [default: Charge Density]
    --t-offset=T        Set time offset in fs. [default: 0].
    --units=U           Set the colorbar units [default: e/cc ]
    --laser             Plot contours of the laser poyting vector.
    --intensity=I -I I  Make a contour of this intensity [default: 3e18]
    --linthresh=L       Set the linear threshold for SymLogPlot [default: 1.0]
    --linscale=L        Set the linear threshold for SymLogPlot [default: 1.0]
    --rotate -R         Rotate instead of flipping x and y (ie., flip x axis).
    --cmap=CMAP         Set the colormap. [default: viridis]

'''
from docopt import docopt;
import numpy as np;
import numpy.linalg as lin;
from pys import parse_ftuple, parse_ituple;
from lspreader.flds import read_indexed, restrict
from lspplot.sclr import S;
from lspplot.pc import pc,highlight;
from lspplot.physics import c,mu0,e0,e;
opts = docopt(__doc__,help=True);
if opts['--nozip']:
    gzip = False;
elif opts['--zip']:
    gzip = True;
else:
    gzip = 'guess';
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
dx = (x[1,0]-x[0,0])*1e-6;
dy = (y[0,1]-y[0,0])*1e-6;

rho = ( np.gradient(Ex,dx,axis=0) + np.gradient(Ey,dy,axis=1) ) * e0 / e * 1e-4;

#####################################
#plotting

#getting options from user
mn,mx = parse_ftuple(opts['--lims'],length=2);
if opts['--rotate']:
    rot,flip = True, False;
else:
    rot,flip = False, True;


#plot the density
toff = float(opts['--t-offset']);
title="{}\nTime: {:.2f} fs".format(titlestr,t*1e6+toff);
r=pc(
    rho,(x,y), lims=(mn,mx),log=opts['--log10'],
    clabel=units, title=title,
    agg=not opts['--show'],
    linthresh=float(opts['--linthresh']),
    linscale=float(opts['--linscale']),
    rotate=rot,
    flip=flip,
    cmap=opts['--cmap'],);


#if opts['--highlight']:
#    myhi  = float(opts['--highlight']);
#    highlight(
#        r, myhi,
#        color="lightyellow", alpha=0.5);

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


