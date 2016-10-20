#!/usr/bin/env python2
'''
For plotting sclr/flds files.
'''
from docopt import docopt;
from lspreader import misc,read;
from lspreader import flds as fldsm;
from lspreader.lspreader import get_header;
import numpy as np;
import numpy.linalg as lin;
from pys import parse_ftuple,test,takef,mk_getkw;
from consts import *

pc_defaults = dict(
    xlabel='microns',
    ylabel='microns',
    title='',
    clabel='',
    cmap='viridis',
    linthresh=1.0,
    linscale=1.0,
)


def pc(q,p=None,**kw):
    '''
    My (easier) pcolormesh.

    Arguments:
      q   -- the quantity.
      p   -- tuple of spatial positions. Optional. If 
             None, plot by index. Here, x is the "polarization"
             direction and y is the "transverse" direction.
             Thus, they are flipped of that of the *cartesian*
             axes of the array. Can be shaped as q as in pcolormesh.
             For 1D arrays, we meshgrid them. Otherwise, we just pass
             them to pcolormesh.

    Keywords Arguments:
      axes      -- use these Axes from matplotlib
      agg       -- use agg
      lims      -- set vlims as a 2-tuple. If not set, or a lim is None,
                   vlims are not set for pcolormesh
      log       -- use log norm. If vmin is negative, use SymLogNorm.
      linthresh -- use this as a value for the linear threshold for
                   SymLogNorm. See the manual for SymLogNorm.
      linscale  -- use this as a value for the linear scale for
                   SymLogNorm. See the manual for SymLogNorm.
      xlabel -- set xlabel
      ylabel -- set ylabel
      title  -- set title
      clabel -- set colorbar label

    Returns:
      A dictionary with axes, pcolormesh object,
      amongst other things. Pass this dict to `highlight`
      and `trajectories` to plot on the same figure.
    '''
    def getkw(l):
        if test(kw,l):
            return kw[l];
        return pc_defaults[l];
    from matplotlib.colors import LogNorm,SymLogNorm;
    import matplotlib;
    if test(kw,"agg"):
        matplotlib.use("agg");
    import matplotlib.pyplot as plt;
    if not test(kw,"axes"):
        kw['axes'] = plt.axes();
    ret={};
    ax = ret['axes'] = kw['axes'];
    if test(kw, 'lims'):
        mn, mx = kw['lims'];
    else:
        mn, mx = None, None
    if test(kw,'log'):
        if mn<0 or test(kw,"force_symlog"):
            linthresh = getkw('linthresh');
            norm = SymLogNorm(
                linthresh=linthresh,
                linscale=getkw('linscale'),
                vmin=mn,vmax=mx);
        else:
            norm= LogNorm();
            q  += 1;
    else:
        norm= None;
    if p == None:
        p = np.arange(q.shape[0]), np.arange(q.shape[1]);
    x,y=p;
    if len(x.shape)==len(y.shape) and len(y.shape)==1:
        y,x = np.meshgrid(y,x,indexing='ij');
    ret['q'] = q;
    ret['x'],ret['y'] = x,y;
    mypc = ret['pc'] =ax.pcolormesh(
        y,x,q,vmin=mn,vmax=mx,cmap=getkw('cmap'),norm=norm);

    if type(norm) is SymLogNorm:
        mnl = int(np.floor(np.log10(-mn)));
        mxl = int(np.floor(np.log10( mx)));
        thrl= int(np.floor(np.log10(np.abs(linthresh))));
        ticks=( [  -10.0**x for x in np.arange(mnl,-thrl-1,-1)]
                + [  0.0 ]
                + [ 10.0**x for x in np.arange(-thrl, mxl+1)] )
        cbar = plt.colorbar(mypc,ticks=ticks);
    else:
        cbar = plt.colorbar(mypc);
    ret['cbar']=cbar;
    if "clabel" in kw and kw["clabel"] is False:
        pass;
    else:
        cbar.set_label(getkw("clabel"));
    ax.set_xlabel(getkw("xlabel"));
    ax.set_ylabel(getkw("ylabel"));
    ax.set_title(getkw("title"));
    return ret;

def highlight(ret, val,
              q=None, color='white', alpha=0.15):
    '''
    Highlight a pc. Essentially a wrapper of plt.contour

    Arguments:
      ret   -- dict returned from pc.
      val   -- value to highlight
      q     -- quantity to highlight. If None, highlight ret's quantity

    Keyword Arguments:
      color -- color of highlight
      alpha -- alpha of highlight

    Returns:
      ret but with stuff that plt.contour adds.
    '''

    ax = ret['axes'];
    if q is None:
        q = ret['q'];
    if not test(ret, 'cbar'):
        ret['cbar'] = plt.colorbar(ret['pc']);
    cbar = ret['cbar'];
    if not test(ret, 'cts'):
        ret['cts'] = [];
    ct = ax.contour(ret['y'],ret['x'], q, [val],
                    colors=[color], alpha = alpha);
    ret['cts'].append(ct);
    if q is ret['q']:
        cbar.add_lines(ct);
    return ret;

trajdefaults = dict(
    alpha = None,
    coords= ['y','x'],
    color = 'black',
    no_resize=False,
    cmap=None,
    color_quantity=None,
    marker='o',
    size=1,
    lw=0.1,
);
    
def trajectories(ret,trajs,**kw):
    '''
    Draw trajectories on a pc. I will provide better documentation later. For
    hints on valid keyword names, look at lspplot.pc.trajdefaults

    Arguments:
      ret   -- dict returned from pc.
      trajs -- trajectories in the form created from lspreader's pmovie
               scheme.

    Keyword Arguments:
      coords    -- coordinates to plot as list of field names
      no_resize -- avoid resizing the axes which happens if the
                   trajectories fall outside of the current axes.
      lw        -- line width of traj
      color     -- color of traj
      cmap      -- colormap of traj
      color_quantity -- a truly crazy thing. Color quantities by either
                         1) a particular quantity
                         2) a function.
                       If this is a str, assume 1). Otherwise let the
                       color of the traj be color_quantity(itr) where
                       itr is a row in trajs. If none, just plot a line.

    Returns:
      None.
    '''

    getkw=mk_getkw(kw, trajdefaults);
    x,y = getkw("coords");
    if not test(kw, "no_resize"):
        xlim, ylim = ret['axes'].get_xlim(), ret['axes'].get_ylim();
    alpha = getkw('alpha');
    af = alpha;
    if alpha is None:
        af = lambda itr: None;
    elif type(alpha) == float:
        af = lambda itr: alpha;
    if not test(kw,"color_quantity"):
        plotit = lambda itr: ret['axes'].plot(
            itr[x], itr[y],
            lw=getkw('lw'),
            alpha=af(itr),
            c=getkw('color'),);
        pass;
    else:
        cf = getkw('color_quantity');
        if type(cf) == str:
            cf = lambda itr: itr[cf];
        plotit = lambda itr: ret['axes'].scatter(
            itr[x], itr[y],
            c=cf(itr),
            lw=getkw('lw'),
            s=getkw('size'),
            alpha=af(itr),
            cmap=getkw('cmap'));
    
    for itr in np.rollaxis(trajs,1):
        plotit(itr);
    if not test(kw, "no_resize"):
        ret['axes'].set_xlim(xlim);
        ret['axes'].set_ylim(ylim);
