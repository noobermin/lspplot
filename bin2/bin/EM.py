#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Just render EM something.

Usage:
    ./EM.py [options] (--show|-s) <i>
    ./EM.py [options] <i> <outname>

Options:
    --help -h
    --show -s          Show
    --verbose -v       Make some noise.
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
    --linthresh=L      Set the linear threshold for SymLogPlot [default: 1e7]
    --linscale=L       Set the linear threshold for SymLogPlot [default: 1e7]
    --cmap=CMAP        Set the colormap. [default: viridis]
    --equal -E         Make spatial dimensions equal.
    --rotate -R        Rotate instead of flipping x and y (ie., flip x axis).
    --blur=R           Blur with this radius.
    --t-offset=T       Set time offset in fs. [default: 0].
    --traj=F           Plot trajectories from this file. If not used,
                       will not plot trajectories.
    --traj-offset=O    Set the offset and factor to get the relevant
                       trajectory timestep such that i_t = factor*i+offset, where
                       i is the passed index in <i>. The factor and offset are passed
                       to this option in the form of a tuple (factor, offset). If not
                       used, script will search for the closest time in the trajectory
                       file.
    --traj-tail=T      Set the "length" in time of the tail. Accepts "inf" which means for
                       all times, otherwise subtract from the time determined by other
                       options and move back that many steps in fs. [default: inf]
    --traj-n=N         Plot only first N trajectories. Pass a 3-tuple in order to slice.
    --traj-energy      Color the trajectory lines by their energy.
    --traj-mass        Set the rest mass of the pmovie particles. [default: 0.511e6]
    --traj-E-log       Logarithmic color for trajectories.
    --traj-maxE=E      Set the maximum E in eV explicitly. If set, anything above
                       will be cut off.
    --traj-minE=E      Set the minimum E in eV. [default: 1]
    --traj-newfmt      Use the new trajectory format.
    --traj-qinvpow=P   Use an inverse power for the charge to alpha instead of
                       linear easing. [default: 2.0]
'''
from docopt import docopt;
import numpy as np;
import numpy.linalg as lin;
from pys import parse_ftuple, parse_ituple, parse_ctuple, parse_stuple;
from pys import fltrx_s, srx_s, rgbrx_s, quote_subs
from lspreader.flds import read_indexed, restrict
from lspplot.sclr import S, E_energy,B_energy,EM_energy, vector_norm, smooth2Dp;
from lspplot.pc import pc, highlight, trajectories;
from lspplot.physics import c,mu0,e0;
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
def mkread(s,N):
    '''lambda's aren't great.'''
    def _f(d):
        return d[s]*N;
    return _f;
quantities.update({ '{}{}'.format(field,comp):dict(
    fvar = [field],
    title='{} field {} component'.format(field,comp),
    units=unit,
    read = mkread('{}{}'.format(field,comp),scale))
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
t  = d['t'];
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

#trajectories
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
        tri_start=None;
        if opts['--traj-tail'] != 'inf':
            tail = float(opts['--traj-tail'])*1e-15*1e9;
            tri_start = np.sum((f['time'] <= t-tail).astype(int));
        pn_end=None;
        pn_start=0;
        pn_step=1;
        if opts['--traj-n']:
            if re.match('^[0-9]+$',opts['--traj-n']):
                pn_end=int(opts['--traj-n']);
            else:
                pn_start,pn_end,pn_step=parse_ituple(opts['--traj-n'],length=3);
        if not opts['--traj-newfmt']:
            tr = f['data'][:,pn_start:pn_end:pn_step].T;
        else:
            keys = list(f.keys());
            ps = len(keys) - 1;
            #magic, think about it
            sz = len(keys[0] if keys[0] != 'time' else keys[1])
            fmt ='{{:0{}}}'.format(sz);
            if pn_end is None:
                pn_end = ps;
            else:
                pn_end = min(ps,pn_end);
            tr = np.array([
                f[fmt.format(i)]
                for i in range(pn_start,pn_end,pn_step)])
        #needs to be here before nans occur.
        maxq=np.max(np.abs(tr['q'])[:,0]);
        
        tr = tr[:,tri_start:tri+1];
    if opts['--verbose']:
        print("size of trajectories: {}".format(tr.shape));
        print("final time is {}".format(trt));
        print("with sclr time as {}".format(t));
        print("maxq is {}".format(maxq));
    pass;

####################################
#massaging data
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
if opts['--rotate']:
    rot,flip = True, False;
else:
    rot,flip = False, True;
#plot the density
toff=float(opts['--t-offset']);
title="{}\nTime: {:.2f} fs".format(titlestr,t*1e6+toff);
r=pc(
    q,(x,y), lims=(mn,mx),log=opts['--log10'],
    clabel=units, title=title,
    agg=not opts['--show'],
    linthresh=float(opts['--linthresh']),
    linscale=float(opts['--linscale']),
    flip=flip,
    rotate=rot,
    cmap=opts['--cmap'],);

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
gm = lambda itr: np.sqrt(itr['ux']**2+itr['uy']**2+itr['uz']**2+1);
massE = float(opts['--traj-mass']);
if opts['--traj']:
    if opts['--traj-energy']:
        print("warning: this doesn't work with alpha settings...");
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
        alphaf=None;
        cmap='copper';
    else:
        cf = None;
        qread = lambda itr: np.abs(np.nan_to_num(itr['q'][0]));
        if opts['--traj-qinvpow']:
            p=float(opts['--traj-qinvpow']);
            alphaf = lambda itr: (qread(itr)/maxq)**(1.0/p)
        else:
            alphaf = lambda itr: qread(itr)/maxq
        cmap=None
    trajectories(
        r, tr,
        alpha=alphaf,
        lw=1,
        coords = [ylabel,'x'],
        cmap   = cmap,
        scale  = (1e4,1e4),
        color_quantity=cf);

import matplotlib.pyplot as plt;
if opts['--equal']:
    plt.axis('equal');
    r['axes'].autoscale(tight=True);
if opts['--show']:
    plt.show();
else:
    plt.savefig(opts['<outname>']);
