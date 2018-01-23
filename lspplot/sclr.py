#!/usr/bin/env python
'''
For plotting sclr/flds files.
'''
from lspreader import read;
from lspreader import flds as fldsm;
from lspreader.lspreader import get_header;
from scipy.signal import convolve;
import numpy as np;
import numpy.linalg as lin;
from pys import parse_ftuple,test,takef;
from lspplot.physics import c,e0,mu0;

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

def twodme(x):
    if type(x) == float:
        x = np.array([x,x]);
    return x;
def smooth2Dp(q, p, s, w,
              type='gauss',
              mode='valid',
              clip=True):
    if len(q.shape)>2:
        raise ValueError("Only for 2D");
    x,y = p;
    dx = np.abs(x[1,0]-x[0,0]);
    dy = np.abs(y[0,1]-y[0,0]);
    if type=='gauss':
        X,Y=np.mgrid[-w[0]/2.0:w[0]/2.0:dx,
                     -w[1]/2.0:w[1]/2.0:dy]
        #gaussian kernel, of course
        kern = np.exp(-( (X/s[0])**2 + (Y/s[1])**2)/2.0 );
        kern = kern/np.sum(kern);
    else:
        raise ValueError('Unknown type "{}"'.format(type));
    if mode!='valid':
        print("warning: use modes other than 'valid' at your own risk");
    ret=convolve(q,kern,mode=mode);
    #someone tell me why
    if clip: ret[ret<0]=0;
    offx=(x.shape[0]-ret.shape[0])/2;
    offy=(y.shape[1]-ret.shape[1])/2;
    return ret, x[offx:-offx,offy:-offy], y[offx:-offx,offy:-offy];

                 
def smooth2D(d,l,
             s=1e-4,w=6e-4,
             type='gauss',
             mode='valid',
             clip=True):
    s=twodme(s);
    w=twodme(w);
    yl =  'y' if 'y' in d else 'z';
    return smooth2Dp(
        d[l], (d['x'], d[yl]), s, w,
        type=type,mode=mode, clip=True);
    


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
