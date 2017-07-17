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
    --quantity=Q -Q Q  Render this quantity. Quantities are E_norm, E_energy,
                       B_norm, B_energy, EM_energy, and S. [default: E_energy]
    --dir=D -D D       Read from this dir [default: .]
    --restrict=R       Restrict it.
    --x-restrict=R     Restrict by positions as a 4 tuple.
    --target=D -t D    Plot contours of this density. If D is a list,
                       plot multiple contours.
    --targetc=C        Set these colors for the contours. If C is a list,
                       plot these colors. [default: darkred]
    --targeta=A        Set the target alphas. If A is a list
                       plot multiple alphas. [default: 0.15]
    --targetq=Q        Set the target quantity. If Q is a list
                       plot multiple quantities. [default: RhoN10]
    --equal -E         Make spatial dimensions equal.
    --t-offset=T       Set time offset in fs. [default: 0].
'''
from docopt import docopt;
import numpy as np;
import numpy.linalg as lin;
from pys import parse_ftuple, parse_ituple, fltrx_s, srx_s, rgbrx, quote_subs
from lspreader.flds import read_indexed, restrict
from lspplot.sclr import S, E_energy,B_energy,EM_energy, vector_norm;
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



quantities = dict(
    E_norm=dict(
        fvar=['E'],
        read= lambda d: vector_norm(d,'E')*1e5,
        title="Electric Field Norm",
        units="V/m"),
    E_energy={
        'fvar':['E'], 'read': E_energy,
        'title': "Electric Field Energy",
        'units': "J / cc"},
    B_norm=dict(
        fvar=['B'],
        read= lambda d: vector_norm(d,'B'),
        title="Magnetic Field Norm",
        units="gauss"),
    B_energy={
        'fvar':['B'], 'read': B_energy,
        'title': "Magnetic Field Energy",
        'units': "J / cc"},
    EM_energy={
        'fvar':['E','B'], 'read': EM_energy,
        'title': "Electromagnetic Field Energy",
        'units': "J / cc"},
    S={
        'fvar':['E','B'], 'read': S,
        'title': "Poynting Vector Norm",
        'units': "W / cm$^2$"},
)
if quantity not in quantities:
    print("quantity is not one of {}".format(quantities.keys()));
    quit();
fvar=quantities[quantity]['fvar']
read=quantities[quantity]['read']
titlestr=quantities[quantity]['title']
units=quantities[quantity]['units']
targetq = opts['--targetq'];
svar=[targetq] if opts['--target'] else None;
#####################################
#reading data
d = read_indexed(int(opts['<i>']),
    flds=fvar,sclr=svar,
    gzip=gzip,dir=opts['--dir'],
              gettime=True,vector_norms=False);
#choosing positions
ylabel =  'z' if np.isclose(d['y'].max(),d['y'].min()) else 'y';
if opts['--x-restrict']:
    res = parse_ftuple(opts['--x-restrict'], length=4);
    res[:2] = [ np.abs(d['x'][:,0]*1e4 - ires).argmin() for ires in res[:2] ];
    res[2:] = [ np.abs(d[ylabel][0,:]*1e4 - ires).argmin() for ires in res[2:] ];
    #including the edges
    res[1]+=1;
    res[3]+=1;
    restrict(d,res);
elif opts['--restrict']:
    res = parse_ituple(opts['--restrict'],length=None);
    restrict(d,res);

#massaging data
t  = d['t'];
x,y = d['x']*1e4,d[ylabel]*1e4
q = read(d);

#####################################
#plotting

#getting options from user
mn,mx = parse_ftuple(opts['--lims'],length=2);
myhi  = float(opts['--highlight']);

#plot the density
toff=float(opts['--t-offset']);
title="{}\nTime: {:.2f} fs".format(titlestr,t*1e6+toff);
r=pc(
    q,(x,y), lims=(mx,mn),log=opts['--log10'],
    clabel=units, title=title,
    agg=True);
highlight(
    r, myhi,
    color="lightyellow", alpha=0.5);

def parse_slist(s):
    
if opts['--target']:
    def parseit(s,rx,parsef=parse_ftuple,quote=False):
        if re.search(rx, s):
            if quote:
                s = quote_subs(s);
            return [eval(s)]
        else:
            return parsef(s, length=None);
    
    if opts['--target'] == 'True':
        H = [1.7e21];
    else:
        H = parseit(opts['--target'], fltrx);

    rgbrx_s = r"\( *(?:{numrx} *, *){{{rep1}}}{numrx} *,{{0,1}} *\)".format(
        rep1=2,
        numrx=fltrx_s);
    crx_s = "(?:{srx}|{rgbrx})".format(srx=srx_s,rgbrx_s);
    
    C = parseit(crx_s, opts['--targetc'],parsef=parse_ctuple, quote=True);
    Q = parseit(srx_s, opts['--targetq'],parsef=parse_stuple, quote=True);
    A = parseit(fltrx_s, opts['--targeta']);

    def lenl(I):
        if len(I) == 1:
            I = I*len(H)
        return I;
    C = lenl(C);
    Q = lenl(I);
    A = lenl(A);
    for h,c,q,a for zip(H,C,Q,A):
        ne = d[q];
        highlight(r, H, q=ne, color=c, alpha=a);
import matplotlib.pyplot as plt;
if opts['--equal']:
    plt.axis('equal');
    r['axes'].autoscale(tight=True);
if opts['--show']:
    plt.show();
else:
    plt.savefig(opts['<outname>']);


