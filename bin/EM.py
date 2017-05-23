#!/usr/bin/env python2
# -*- coding: utf-8 -*-
'''
Just render EM something.

Usage:
    ./EM.py [options] (--show|-s) <i>
    ./EM.py [options] <i> <outname>

Options:
    --help -h
    --show -s          Show
    --nozip -U         flds are NOT gzipped.
    --zip   -Z         flds are gzipped. If neither of these two are set,
                       guess based on name.
    --log10 -l         Log it.
    --lims=LIM         Set lims [default: (1e2,6e8)]
    --highlight=H      Set highlight [default: 3e8]
    --quantity=Q -Q Q  Render this quantity [default: E_energy]
    --dir=D -D D       Read from this dir [default: .]
    --restrict=R       Restrict it.
    --target -t        Plot contours of the target.
'''
from docopt import docopt;
import numpy as np;
import numpy.linalg as lin;
from pys import parse_ftuple, parse_ituple;
from lspreader.flds import read_indexed, restrict
from lspplot.sclr import S, E_energy,B_energy,EM_energy;
from lspplot.pc import pc, highlight;
from lspplot.consts import c,mu0,e0;

opts = docopt(__doc__,help=True);
if opts['--nozip']:
    gzip = False;
elif opts['--zip']:
    gzip = True;
else:
    gzip = 'guess';
quantity = opts['--quantity'];

quantities = {
    'E_energy':{
        'fvar':['E'], 'read': E_energy,
        'title': "Electric Field Energy",
        'units': "J / cc"},
    'B_energy': {
        'fvar':['B'], 'read': B_energy,
        'title': "Magnetic Field Energy",
        'units': "J / cc"},
    'EM_energy': {
        'fvar':['E','B'], 'read': EM_energy,
        'title': "Electromagnetic Field Energy",
        'units': "J / cc"},
    'S': {
        'fvar':['E','B'], 'read': S,
        'title': "Poynting Vector Norm",
        'units': "W / cm$^2$"},
};
if quantity not in quantities:
    print("quantity is not one of {}".format(quantities.keys()));
    quit();
fvar=quantities[quantity]['fvar']
read=quantities[quantity]['read']
titlestr=quantities[quantity]['title']
units=quantities[quantity]['units']
svar=['RhoN10'] if opts['--target'] else None;
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
if np.isclose(y.max(),y.min()):
    y = d['z']*1e4
q = read(d);

#####################################
#plotting

#getting options from user
mn,mx = parse_ftuple(opts['--lims'],length=2);
myhi  = float(opts['--highlight']);

#plot the density
title="{}\nTime: {:.2f} fs".format(titlestr,t*1e6);
r=pc(
    q,(x,y), lims=(mx,mn),log=opts['--log10'],
    clabel=units, title=title,
    agg=True);
highlight(
    r, myhi,
    color="lightyellow", alpha=0.5);

if opts['--target']:
    ne = d['RhoN10'];
    highlight(
        r,1e18, q=ne,
        color="red", alpha=0.15);
    highlight(
        r, 1.7e21, q=ne,
        color="red", alpha=0.15);

import matplotlib.pyplot as plt;
if opts['--show']:
    plt.show();
else:
    plt.savefig(opts['<outname>']);


