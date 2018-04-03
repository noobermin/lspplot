#!/usr/bin/env python
'''
For plotting sclr/flds files.
'''
from lspreader import read;
from lspreader.flds import vector_norm, getvector, restrict
from lspreader.lspreader import get_header;
from scipy.signal import convolve;
import numpy as np;
import numpy.linalg as lin;
from pys import parse_ftuple,test,takef;
from lspplot.physics import c,e0,mu0;


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
    offx=(x.shape[0]-ret.shape[0])//2;
    offy=(y.shape[1]-ret.shape[1])//2;
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

def _axis(i):
    dims=['x','y','z']
    if type(i) == str:
        return (dims.index(i),i);
    return i,dims[i];
def flatten3d_aa(d, q=None, coord=0.0, dx=1e-4, axis='z',**kw):
    '''
    Flatten 3d arrays which along an axis. Averages over
    a width of dx.
    '''
    if type(axis) != tuple: axis  = _axis(axis);
    i,axis = axis[:2];
    good = d[axis] <= coord + dx/2.0;
    good&= d[axis] >= coord - dx/2.0;
    if type(q) == str:
        return np.average(d[q][good], axis=i);
    shape = list(good.shape)
    shape[i] = -1;
    shape = tuple(shape);
    if q is None:
        for k in d:
            if k=='t': continue;
            d[k] = np.average(d[k][good].reshape(shape),axis=i);
        return d;
    else:
        return [np.average(d[iq][good].shape(shape), axis=i) for iq in q];


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
