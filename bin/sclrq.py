#!/usr/bin/env python2
# -*- coding: utf-8 -*-
'''
Just render something.

Usage:
    ./sclrq.py [options] (--show|-s) <i>
    ./sclrq.py [options] <i> <outname>

Options:
    --help -h
    --show -s           Show
    --nozip -U          flds are NOT gzipped.
    --zip   -Z          flds are gzipped. If neither of these two are set,
                        guess based on name.
    --log10 -l          Log it.
    --lims=LIM          Set lims [default: (1e18,1e23)]
    --highlight=H       Set highlight.
    --quantity=Q -Q Q   Render this quantity [default: RhoN10]
    --dir=D -D D        Read from this dir [default: .]
    --restrict=R        Restrict it.
    --x-restrict=R      Restrict by positions as a 4 tuple.
    --title=T           Set the title [default: Electron density]
    --units=U           Set the colorbar units [default: number/cc]
    --laser             Plot contours of the laser poyting vector.
    --intensity=I -I I  Make a contour of this intensity [default: 3e18]
    --equal -E          Make spatial dimensions equal.
    --no-ticks          Don't include ticks.
    --orientation=V     "V" for vertical or "H" for horizontal [default: V]
'''
from docopt import docopt;
import numpy as np;
import numpy.linalg as lin;
from pys import parse_ftuple, parse_ituple;
from lspreader.flds import read_indexed, restrict
from lspplot.sclr import S;
from lspplot.pc import pc,highlight;
from lspplot.consts import c,mu0,e0;

opts = docopt(__doc__,help=True);

quantity = opts['--quantity'];
fvar=['E','B'] if opts['--laser'] else None;
titlestr=opts['--title']
units=opts['--units'];
svar=[quantity];
if opts['--nozip']:
    gzip = False;
elif opts['--zip']:
    gzip = True;
else:
    gzip = 'guess';
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

x,y = d['x']*1e4, d[ylabel]*1e4;
#massaging data
t  = d['t'];
q = d[quantity];

#####################################
#plotting

#getting options from user
mn,mx = parse_ftuple(opts['--lims'],length=2);


#plot the density
title="{}\nTime: {:.2f} fs".format(titlestr,t*1e6);

#orientation of colorbar
if opts['--orientation'] == "V":
    orient = "vertical"
elif opts['--orientation'] == "H":
    orient = "horizontal"
else:
    print('orientation must be either "V" or "H"');
    print(__doc__);
    quit();

r=pc(
    q,(x,y), lims=(mn,mx),log=opts['--log10'],
    clabel=units, title=title,
    agg=not opts['--show'],
    orient=orient,);

if opts['--highlight'] and (opts['--hightlight'] != "None" and opts['--highlight'] != 'none'):
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


