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
    --highlight=H      Set highlight for field quantity.
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
    --blur=R           Blur with this radius.
    --t-offset=T       Set time offset in fs. [default: 0].
'''
from docopt import docopt;
import numpy as np;
import numpy.linalg as lin;
from pys import parse_ftuple, parse_ituple, parse_ctuple, parse_stuple;
from pys import fltrx_s, srx_s, rgbrx_s, quote_subs
from lspreader.flds import read_indexed, restrict
from lspplot.sclr import S, E_energy,B_energy,EM_energy, vector_norm, smooth2Dp;
from lspplot.pc import pc, highlight;
from lspplot.consts import c,mu0,e0;

import re;

opts = docopt(__doc__,help=True);
if opts['--nozip']:
    gzip = False;
elif opts['--zip']:
    gzip = True;
else:
    gzip = 'guess';

def parseit(s,rx=fltrx_s,parsef=parse_ftuple,quote=False):
    ''' helper for parsing tuples '''
    #this needs to be match! not search to not match parens.
    if re.match(rx, s):
        if quote:
            s = quote_subs(s);
        return [eval(s)]
    else:
        return parsef(s, length=None);
    

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
quantities.update({ '{}{}'.format(field,comp):dict(
    fvar = [field],
    title='{} field {} component'.format(field,comp),
    units=unit,
    read = lambda d: d['{}{}'.format(field,comp)]*scale,)
  for field,unit,scale in [('E','V/m',1e5),('B','gauss',1.0)]
  for comp in ['x','y','z'] });

if quantity not in quantities:
    print("quantity is not one of {}".format(quantities.keys()));
    quit();
fvar=quantities[quantity]['fvar']
read=quantities[quantity]['read']
titlestr=quantities[quantity]['title']
units=quantities[quantity]['units']

if opts['--target']:
    svar = parseit(opts['--targetq'], rx=srx_s,parsef=parse_stuple,quote=True)
    #unique
    svar = list(set(svar));
else:
    svar = None;
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

if opts['--blur']:
    rad = float(opts['--blur']);
    w=rad*6;
    q,x,y = smooth2Dp(
        q, (x,y), [0.1,0.1], [0.6,0.6]);

#####################################
#plotting

#getting options from user
mn,mx = parse_ftuple(opts['--lims'],length=2);

#plot the density
toff=float(opts['--t-offset']);
title="{}\nTime: {:.2f} fs".format(titlestr,t*1e6+toff);
r=pc(
    q,(x,y), lims=(mx,mn),log=opts['--log10'],
    clabel=units, title=title,
    agg=True);
if opts['--highlight']:
    highlight(
        r, float(opts['--highlight']),
        color="lightyellow", alpha=0.5);

    
if opts['--target']:    
    if opts['--target'] == 'True':
        H = [1.7e21];
    else:
        H = parseit(opts['--target']);
    rgbrx_s = r"\( *(?:{numrx} *, *){{{rep1}}}{numrx} *,{{0,1}} *\)".format(
        rep1=2,
        numrx=fltrx_s);
    crx_s = "(?:{srx}|{rgbrx})".format(srx=srx_s,rgbrx=rgbrx_s);
    
    C = parseit(opts['--targetc'], rx=crx_s, parsef=parse_ctuple, quote=True);
    Q = parseit(opts['--targetq'], rx=srx_s, parsef=parse_stuple, quote=True);
    A = parseit(opts['--targeta']);
    def lenl(I):
        if len(I) == 1:
            I = I*len(H);
        return I;
    C = lenl(C);
    Q = lenl(Q);
    A = lenl(A);
    for h,c,l,a in zip(H,C,Q,A):
        cq=d[l];
        if opts['--blur']:
            offx=(d[l].shape[0]-q.shape[0])/2
            offy=(d[l].shape[1]-q.shape[1])/2
            cq = cq[offx:-offx,offy:-offy];
        highlight(r, h, q=cq, color=c, alpha=a);
import matplotlib.pyplot as plt;
if opts['--equal']:
    plt.axis('equal');
    r['axes'].autoscale(tight=True);
if opts['--show']:
    plt.show();
else:
    plt.savefig(opts['<outname>']);
