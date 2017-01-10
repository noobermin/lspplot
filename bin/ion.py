#!/usr/bin/env python2
# -*- coding: utf-8 -*-
'''
Just render ion something.

Usage:
    ./ion.py [options] (--show|-s) <i>
    ./ion.py [options] <i> <outname>

Options:
    --help -h
    --show -s           Show
    --nozip -U          flds are NOT gzipped.
    --log10 -l          Log it.
    --lims=LIM          Set lims [default: (1e18,1e23)]
    --highlight=H       Set highlight.
    --ions=Q -Q Q       Render the sum of these quantities as a tuple of
                        items. [default: (RhoN9,RhoN8,RhoN7,RhoN6,RhoN5,RhoN4,RhoN3,RhoN2,RhoN11)]
    --charges=Q -q Q    Use these weights for each of the quantities
                        items. [default: (8.0,7.0,6.0,5.0,4.0,3.0,2.0,1.0,1.0)]
    --average=A -A      Average over space by these many cells, one for each dimension, and
                        up and down. That is, (1,0) averages by x +1 and -1. [default: (0,0)]
    --dir=D -D D        Read from this dir [default: .]
    --restrict=R        Restrict it.
    --x-restrict=R      Restrict by positions as a 4 tuple.
    --title=T           Set the title [default: Ion Charge Density]
    --units=U           Set the colorbar units [default: e/cc]
    --laser             Plot contours of the laser poyting vector.
    --intensity=I -I I  Make a contour of this intensity [default: 3e18]
    --linthresh=L       Set the linear threshold for SymLogPlot [default: 1.0]
    --linscale=L        Set the linear threshold for SymLogPlot [default: 1.0]
    --cmap=CMAP         Set the colormap. [default: viridis]
    --equal -E          Make spatial dimensions equal.
    --no-ticks          Don't include ticks.
'''
from docopt import docopt;
import numpy as np;
import numpy.linalg as lin;
from pys import parse_ftuple, parse_ituple;
from lspreader.flds import read_indexed, restrict
from lspplot.sclr import S;
from lspplot.pc import pc,highlight;
from lspplot.consts import c,mu0,e0;
import re
opts = docopt(__doc__,help=True);
gzip = not opts['--nozip'];
if not re.match(r" *[\[\(](?:\w+,)*(?:\w+)?[\[\)]", opts['--ions']):
    raise ValueError("--ions argument invalid, see --help");
charges=parse_ftuple(opts['--charges'],length=None);
qs = re.search("[\[\(](.+)[\]\)]",opts['--ions']).group(1).split(",");
if qs[-1] == '': qs = qs[:-1];
if len(qs) != len(charges):
    raise ValueError(
        "--charges and --ions arguments have differing lengths.");
fvar=['E','B'] if opts['--laser'] else None;
titlestr=opts['--title']
units=opts['--units'];
#####################################
#reading data
d = read_indexed(int(opts['<i>']),
    flds=fvar,sclr=qs,
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
x,y = d['x']*1e4, d[ylabel]*1e4;
avg = parse_ituple(opts['--average'],length=2);
#massaging data
t  = d['t'];
q = np.sum([d[Q]*i for Q,i in zip(qs,charges)],axis=0);
#averaging
acc = [q];
qp = np.pad(q, ((avg[0],)*2, (avg[1],)*2), 'edge');
qps = [ qp[:,avg[1]:-avg[1]],
        qp[avg[0]:-avg[0],:],];
for I,(ia,iqp) in enumerate(zip(avg,qps)):
    l=2*ia + 1;
    if l == 1: continue;
    Is = [
        [(None,None)]*l, [(None,None)]*l,
    ];
    st  = [None] + list(range(1,l));
    end = list(range(-l+1,0)) + [None];
    del st[ia];
    del end[ia];
    Is[I] = [
        (ist,iend) for ist,iend in zip(st,end)
    ]
    acc += [ iqp[Is[0][i][0]:Is[0][i][1],
                 Is[1][i][0]:Is[1][i][1]]
             for i in xrange(ia)
    ];
q = np.average(acc,axis=0);
#####################################
#plotting

#getting options from user
mn,mx = parse_ftuple(opts['--lims'],length=2);

#plot the density
title="{}\nTime: {:.2f} fs".format(titlestr,t*1e6);
r=pc(
    q,(x,y), lims=(mn,mx),log=opts['--log10'],
    clabel=units, title=title,
    agg=not opts['--show'],
    linthresh=float(opts['--linthresh']),
    linscale=float(opts['--linscale']),
    cmap=opts['--cmap'],
);

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
if opts['--equal']:
    plt.axis('equal');
    r['axes'].autoscale(tight=True);
if opts['--no-ticks']:
    plt.tick_params(
        axis='both',
        which='both',
        bottom='off',
        top='off',
        right='off',
        left='off');
    
if opts['--show']:
    plt.show();
else:
    plt.savefig(opts['<outname>']);


