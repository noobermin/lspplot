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
    --traj-energy       Color the trajectory lines by their energy.
    --traj-E-log        Logarithmic color for trajectories.
    --traj-maxE=E       Set the maximum E in eV explicitly. If set, anything above
                        will be cut off.
    --traj-minE=E       Set the minimum E in eV. [default: 1]
'''
from docopt import docopt;
import numpy as np;
import numpy.linalg as lin;
from pys import parse_ftuple, parse_ituple;
from lspreader.flds import read_indexed, restrict
from lspplot.sclr import S;
from lspplot.pc import pc,highlight,trajectories;
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
        print("with sclr time as {}".format(t));
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
gm = lambda itr: np.sqrt(itr['ux']**2+itr['uy']**2+itr['uz']**2+1);
massE = .511e6
if opts['--traj']:
    tr[coords[1]]*=1e4;
    tr[coords[0]]*=1e4;
    if opts['--traj-energy']:
        en = lambda itr:np.nan_to_num(massE*(gm(itr)-1));
        if opts['--traj-E-log']:
            minE = float(opts['--traj-minE']);
            def _energy(itr):
                E = en(itr);
                return np.log10(
                    np.where(E < minE, minE, E));
            energy=_energy;
        else:
            energy=en;
        #find max energy
        if opts['--traj-maxE']:
            maxE=float(opts['--traj-maxE']);
            def _cf(itr):
                E = energy(itr);
                return np.where(E>maxE, 1.0, E/maxE)
            cf = _cf;
        else:
            maxE=np.max(energy(tr));
            cf = lambda itr: energy(itr)/maxE;
    else:
        cf = None;
    #massaging alpha
    maxq=np.log10(np.max(np.abs(tr['q'])[0,:]));
    alphaf = lambda itr: np.log10(np.abs(itr['q'])[0])/maxq
    trajectories(
        r, tr,
        alpha=alphaf,
        lw=0,
        coords = list(reversed(coords)),
        cmap   = 'copper',
        color_quantity=cf);
if opts['--show']:
    plt.show();
else:
    plt.savefig(opts['<outname>']);


