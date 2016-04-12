#!/usr/bin/env python2
'''
For plotting sclr/flds files.
'''
from lspreader import read;
from lspreader import flds as fldsm;
from lspreader.lspreader import get_header;
import numpy as np;
import numpy.linalg as lin;
from pys import parse_ftuple,test,takef;
from consts import *

def getvector(d,s):
    return np.array([d[s+"x"],d[s+"y"],d[s+"z"]]);
def vector_norm(d,k):
    return lin.norm(getvector(d,k),axis=0)

def readfiles(i,flds=None,sclr=None,
              gzip=True, dir='.',vector_norms=True,
              keep_xs=False,
              gettime=False):
    
    fldsname = '{}/flds{}.p4{}'.format(
        dir, i, '.gz' if gzip else '');
    sclrname = '{}/sclr{}.p4{}'.format(
        dir, i, '.gz' if gzip else '');
    if not (flds or sclr):
        raise ValueError("Must specify flds or sclr to read.");
    elif flds is not None and sclr is not None:
        sd=read(sclrname,
                var=sclr,remove_edges=True, gzip=gzip);
        fd=read(fldsname,
                var=flds,remove_edges=True, gzip=gzip);
        srt = fldsm.firstsort(sd);
        sd = fldsm.sort(sd,srt);
        sd = fldsm.squeeze(sd);
        fd = fldsm.sort(fd,srt);
        fd = fldsm.squeeze(fd);
        ret = dict(sd=sd,fd=fd);
        ret.update({k:sd[k] for k in sd});
        ret.update({k:fd[k] for k in fd});
        if vector_norms:
            ret.update({k:vector_norm(ret,k) for k in flds})
        if gettime:
            ret['t'] = get_header(sclrname,gzip=gzip)['timestamp'];
    else:
        if flds:
            var = flds;
            name= fldsname;
            key = 'fd';
        else:
            var = sclr;
            name= sclrname;
            key = 'sd';
        ret=read(name,var=var,remove_edges=True,gzip=gzip);
        srt = fldsm.firstsort(ret);
        ret = fldsm.sort(ret,srt);
        ret = fldsm.squeeze(ret);
        if flds and vector_norms:
            ret.update({k:vector_norm(ret,k) for k in flds})
        if gettime:
            ret['t'] = get_header(name,gzip=gzip)['timestamp'];
    if not keep_xs:
        ret.pop('xs',None);
        ret.pop('ys',None);
        ret.pop('zs',None);
    return ret;

def restrict(d,restrict):
    notqs = ['t','xs','ys','zs','fd','sd']
    keys  = [k for k in d if k not in notqs];
    if len(restrict) == 2:
        for k in keys:
            d[k] = d[k][restrict[0]:restrict[1]]
    elif len(restrict) == 4:
        for k in keys:
            d[k] = d[k][
                restrict[0]:restrict[1],
                restrict[2]:restrict[3]
            ];
    elif len(restrict) == 6:
        for k in keys:
            d[k] = d[k][
                restrict[0]:restrict[1],
                restrict[2]:restrict[3],
                restrict[4]:restrict[5]
            ];
    else:
        raise ValueError("restrict of length {} is not valid".format(
            len(restrict)));


def E_energy(d):
    return e0*(vector_norm(d,'E')*1e5)**2/2.0*1e-6;
def B_energy(d):
    return (vector_norm(d,'B')*1e-4)**2/(mu0*2.0)*1e-6;
def EM_energy(d):
    return (E_energy(d) + B_energy(d))
def S(d):
    E = getvector(d,'E');
    B = getvector(d,'B');
    return lin.norm(np.cross(E*1e5,B*1e-4,axis=0),axis=0)/mu0*1e-4;
