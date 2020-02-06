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



def ndme(x,n=2):
    if type(x) == float:
        x = np.array([x]*n);
    return x;


def smooth(q, p, s, w,
           type='gauss',
           mode='valid',
           clip=True,
           zeroshift=True):
    N = len(q.shape);
    s = ndme(s, n=N);
    w = ndme(w, n=N);
    dxs = [ np.abs(ix[1]-ix[0]) for ix in p ];
    qmn = 0;
    if zeroshift and q.min() < 0.0:
        qmn = q.min();
        q -= qmn;
    if type=='gauss':
        P = [ 
            np.arange(-iw/2.0, iw/2.0+idx, idx) for iw,idx in zip(w, dxs)
        ]
        P = np.meshgrid(*P, indexing='ij');
        arg = np.sum([ 0.5*(X/si)**2 for X,si in zip(P,s) ],axis=0);
        kern = np.exp(-arg);
        kern[:] = kern/np.sum(kern);
        #gaussian kernel, of course
    else:
        raise ValueError('Unknown type "{}"'.format(type));
    if mode!='valid':
        print("warning: use modes other than 'valid' at your own risk");
    ret=convolve(q,kern,mode=mode);
    #someone tell me why
    if clip: ret[ret<0]=0;
    def div2(i):
        r = i//2;
        return r,-(r+i%2);
    offs = [ div2(len(ix) - shi) for ix,shi in zip(p, ret.shape) ];
    p = [ ix[st:en] for ix,(st,en) in zip(p,offs) ];
    ret += qmn;
    return ret, p;

def getrestr_aax(ix,ir):
    '''create a slice using the bounding range ir of ix, with ir upper bound is open'''
    out = np.where(np.logical_and(ir[0] <= ix, ix < ir[1]))[0];
    return slice(out[0],out[-1]+1);

def getrestr_aa(ix,ir):
    '''create a slice using the bounding range ir of ix'''
    out = np.where(np.logical_and(ir[0] <= ix, ix <= ir[1]))[0];
    return slice(out[0],out[-1]+1);

def restrict_grid(lims,grid,include_sup=False):
    '''restrict a grid to a cubeoid specified by lims:
         lims ~= (zmin,zmax,ymin,ymax,xmin,xmax)
         grid ~= (zs,ys,xs)
       returns (zs,ys,xs), (kg,jg,ig)
       
       keywords:
       ========
       
       include_sup -- the top of the restricted grid will be
                      the upper edge of the grid (supremum)
    '''
    rs = chunks(lims,2);
    gs = [ getrestr_aa(xs,ir)
           for xs,ir in zip(grid,rs) ];
    def _up(slc): return slice(slc.start,slc.stop+1);
    if include_sup:
        xs = [ xgrid[_up(g)] for g,xgrid in zip(gs,grid) ];
    else:
        xs = [ xgrid[g] for g,xgrid in zip(gs,grid) ];
    return xs, gs;

                 
def smooth2D(d,l,
             s=1e-4,w=6e-4,
             type='gauss',
             mode='valid',
             clip=True):
    s=ndme(s);
    w=ndme(w);
    yl =  'y' if 'y' in d else 'z';
    ret,p = smooth(
        d[l], (d['x'], d[yl]), s, w,
        type=type,mode=mode, clip=True);
    return ret, p[0], p[1];

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


def lspunits(q):
    if "E" in q:
        return 511e5
    elif "B" in q:
        return 1704.5
    elif "J" in q:
        return 1356.4
    elif "RhoN" in q:
        return 2.82396e11;
    if "qp" in q:
        return 4.5245e-2*1e-6;
    else:
        raise ValueError("quantity not known: {}".format(q));
    pass;

def basicstats(d):
    pos = d > 0.0;
    ret = dict(
        shape=d.shape,
        min=np.min(d),
        max=np.max(d),
        avg=np.average(d),
        anynan=np.any(np.isnan(d)),);
    if np.max(pos) == False:
        ret.update(
            avgpos=None,
            minpos=None,);
    else:
        ret.update(
            avgpos=np.average(d[pos]),
            minpos=np.min(d[pos]));
    return ret;

def guess_step(s):
    m=re.search('([0-9]+)[A-Z,a-z,_].*\.np[yz]',s);
    if m: return int(m.group(1));

    
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
