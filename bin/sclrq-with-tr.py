#!/usr/bin/env python2
# -*- coding: utf-8 -*-
'''
Just render something...with trajectories

Usage:
    ./sclrq.py [options] (--show|-s) <i>
    ./sclrq.py [options] <i> <outname>

Options:
    --help -h
    --show -s           Show
    --verbose -v        Make some noise.
    --nozip -U          flds are NOT gzipped.
    --log10 -l          Log it.
    --lims=LIM          Set lims [default: (1e2,6e8)]
    --highlight=H       Set highlight [default: 3e8]
    --quantity=Q -Q Q   Render this quantity [default: RhoN10]
    --dir=D -D D        Read from this dir [default: .]
    --restrict=R        Restrict it.
    --title=T           Set the title [default: Electron density]
    --units=U           Set the colorbar units [default: number/cc]
    --laser             Plot contours of the laser poyting vector.
    --intensity=I -I I  Make a contour of this intensity [default: 3e18]
    --traj=F            Plot trajectories from this file. If not used,
                        will not plot trajectories.
    --traj-offset=O     Set the offset and factor to get the relevant
                        trajectory timestep such that i_t = factor*i+offset, where
                        i is the passed index in <i>. The factor and offset are passed
                        to this option in the form of a tuple (factor, offset). If not
                        used, script will search for the closest time in the trajectory
                        file.
    --traj-n=N          Plot only first N trajectories.
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
gzip = not opts['--nozip'];
quantity = opts['--quantity'];

fvar=['E','B'] if opts['--laser'] else None;
titlestr=opts['--title']
units=opts['--units'];
svar=[quantity];
#####################################
#reading data
i = int(opts['<i>']);
d = read_indexed(i,
    flds=fvar,sclr=svar,
    gzip=gzip,dir=opts['--dir'],
              gettime=True,vector_norms=False);
t  = d['t'];

if opts['--traj']:
    factor, offset = None, None;
    if opts['--traj-offset']:
        factor, offset = parse_ituple(opts['--traj-offset'],length=2);
    with np.load(opts['--traj'], mmap_mode='r') as f:
        if factor:
            tri = i*factor+offset;
            trt = f['time'][tri];
            if not np.isclose(trt,t):
                import sys
                sys.stderr.write(
                    "warning: time from trajectory is {} while time from sclr is {}\n".format(
                        trt,t));
        else:
            tri = np.sum((f['time'] <= t).astype(int));
            trt = f['time'][tri];
        tr = f['data'][:tri+1,:];
    if opts['--traj-n']:
        tr = tr[:,:int(opts['--traj-n'])];
    if opts['--verbose']:
        print("size of trajectories: {}".format(tr.shape));
        print("final time is {}".format(trt));
    pass;
if opts['--restrict']:
    res = parse_ituple(opts['--restrict'],length=None);
    restrict(d,res);

#massaging data
x,y = d['x']*1e4,d['y']*1e4
coords = ['x','y'];
if np.isclose(y.max(),y.min()):
    y = d['z']*1e4
    coords[1] = 'z';
q = d[quantity];

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
    agg=not opts['--show']);
highlight(
    r, myhi,
    color="lightyellow", alpha=0.5);

if opts['--laser']:
    laser = S(d);
    I = float(opts['--intensity']);
    highlight(r, I, q=laser,
              color="red", alpha=0.15);
    
import matplotlib.pyplot as plt;
if opts['--traj']:
    tr[coords[1]]*=1e4;
    tr[coords[0]]*=1e4;
    for itr in np.rollaxis(tr,1):
        plt.plot(itr[coords[1]],itr[coords[0]]);
if opts['--show']:
    plt.show();
else:
    plt.savefig(opts['<outname>']);


