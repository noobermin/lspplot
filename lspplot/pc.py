#!/usr/bin/env python
'''
For plotting sclr/flds files.
'''
from docopt import docopt;
from lspreader import read;
from lspreader import flds as fldsm;
from lspreader.lspreader import get_header;
import numpy as np;
import numpy.linalg as lin;
from pys import parse_ftuple,test,takef,mk_getkw;
from lspplot.physics import c,e0,mu0;
import re;

pc_defaults = dict(
    xlabel='microns',
    ylabel='microns',
    title='',
    clabel='',
    cmap='viridis',
    linthresh=1.0,
    linscale=1.0,
    orient='vertical',
)

def natlabel(v,fmt='1.1e'):
    f,p = np.modf(np.log10(v));
    if np.isclose(f,0.0):
        return f"10$^{{{int(p):d}}}$"
    else:
        return exptolabel(v,fmt);

def exptolabel(v,fmt='1.1e'):
    s = f'{v:{fmt}}'
    s = re.sub("e","$\\\times$10$^{",s);
    s = s + "}$"
    return s;

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
      cbar      -- if set to false, don't output cbar. Default is True.
      xlabel -- set xlabel
      ylabel -- set ylabel
      title  -- set title
      clabel -- set colorbar label
      orient -- the orientation for the colorbar.
      cmap   -- set the colormap.
      norm   -- manually set the norm.
      discrete_10 -- Use levels of log10 for the norm and colormap, making it
                     a discrete color map.
      rotate -- rotate x and y.
      flip   -- flips x and y. mimics behavior before
                version 0.0.12.
      nofloor -- raise error if there are no values > 0.0. Otherwise,
                 proceed but raise the floor on the quantity.
      nocopy  -- copies q so keep from modifying the passed quantity.
                 Unset to save memory but be aware of side-effects.

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
    from matplotlib.colors import BoundaryNorm,LinearSegmentedColormap;
    from matplotlib import cm;
    import matplotlib;
    if test(kw,"agg"):
        matplotlib.use("agg");
    import matplotlib.pyplot as plt;
    if not test(kw,"axes"):
        kw['axes'] = plt.axes();
    ret={};
    ax = ret['axes'] = kw['axes'];
    mn, mx = None, None;
    vmin,vmax = None, None;
    if test(kw, 'lim') and not test(kw,'lims'):
        kw['lims'] = kw['lim'];
    if test(kw, 'lims'): mn, mx = kw['lims'];
    if not test(kw, 'nocopy'):
        q=np.copy(q);
    cmap = getkw('cmap');
    if test(kw, 'norm'):
        norm = kw['norm'];
    elif test(kw,'discrete_10'):
        if mn < 0 or test(kw, "force_symlog"):
            raise NotImplementedError(
                "need to implement symlog for discrete_10");
        mnf,mnw = np.modf(np.log10(mn));
        mxf,mxw = np.modf(np.log10(mx));
        pows   = np.arange(mnw,mxw+1)
        levels = 10**pows;
        levels[0] = mn;
        if not np.isclose(mxf,0.0):
            levels = np.concatenate((levels,[mx]));
        if type(cmap) == str:
            cmap = cm.get_cmap(cmap);
        if len(levels) > cmap.N:
            raise ValueError( (
                f"you want to use discrete_10 with this many log10 levels?"
                f" (len(levels) == {len(levels)}) > (cmap.N == {cmap.N})"))
        di = cmap.N // len(levels);
        #up to the end
        st = cmap.N - di*len(levels);
        colors = [ cmap(i) for i in range(st,cmap.N) ];
        cmap = LinearSegmentedColormap.from_list(
            'Discrete cmap', colors, len(colors));
        norm = BoundaryNorm(levels, cmap.N);
    elif test(kw,'log'):
        if mn is not None and (mn<0 or test(kw,"force_symlog")):
            linthresh = getkw('linthresh');
            norm = SymLogNorm(
                linthresh=linthresh,
                linscale=getkw('linscale'),
                vmin=mn,vmax=mx);
        else:
            norm= LogNorm(vmin=mn,vmax=mx);
            if len(q[ q > 0.0 ]) == 0:
                errmsg="quantity has no values greater than zero with log";
                if test(kw, 'nofloor') or mn is None:
                    raise ValueError(errmsg);
                print("warning: {}".format(errmsg));
                print("setting all values to min");
                q[:] = mn;
            else:
                floor = q[ q > 0.0 ].min();
                if mn is not None:
                    floor = min(mn,floor);
                q[q <= 0.0] = floor;
    else:
        norm= None;
        vmin,vmax=mn,mx;
    if p is None:
        p = np.arange(q.shape[1]), np.arange(q.shape[0]);
    x,y=p;
    ret['x'],ret['y'] = p;
    ret['q']  = q;
    if test(kw, 'flip') or test(kw, 'rotate'):
        q=q.T;
        x,y=y,x;
    ret['flip'] = test(kw, 'flip');
    ret['rotate'] = test(kw, 'rotate');
    #developers developers developers
    if norm is not None:
        ret['pc'] =ax.pcolormesh(x,y,q,cmap=cmap,norm=norm);
    else:
        ret['pc'] =ax.pcolormesh(x,y,q,vmin=vmin,vmax=vmax,cmap=cmap);
    mypc = ret['pc'];
    if ret['rotate']:
        ret['axes'].invert_xaxis();
    if 'cbar' in kw and kw['cbar'] is False:
        ret['cbar'] = cbar = None;
    else:
        ret['cbar'] = cbar = plt.colorbar(
            mypc,orientation=getkw('orient'));
    
    if type(norm) is SymLogNorm:
        mnl = int(np.floor(np.log10(-mn)));
        mxl = int(np.floor(np.log10( mx)));
        thrl= int(np.floor(np.log10(np.abs(linthresh))));
        negpows = np.arange(thrl,mnl+1)[::-1];
        pospows = np.arange(thrl,mxl+1);
        ticks   = np.concatenate( (
            -10.0**negpows, [0.0], 10.0**pospows));
        tlabels = (
            [ "$-$10$^{{{}}}$".format(int(p)) for p in negpows]
            + ['0']
            + ["$+$10$^{{{}}}$".format(int(p)) for p in pospows]);
        #ugh...
        if cbar:
            cbar.set_ticks(ticks);
            cbar.set_ticklabels(tlabels);
    elif type(norm) is BoundaryNorm:
        ticks   = levels
        tlabels = [ natlabel(v) for v in levels ];
        if cbar:
            cbar.set_ticks(ticks);
            cbar.set_ticklabels(tlabels);
    if test(kw,"clabel") and cbar:
        cbar.set_label(getkw("clabel"));
    ax.set_xlabel(getkw("xlabel"));
    ax.set_ylabel(getkw("ylabel"));
    ax.set_title(getkw("title"));
    return ret;

def timelabel(ret, s,loc='lower right',fig=False,**kw):
    '''
    Create a label somewhere. Useful for time.
    
    Arguments:
      ret   -- dict returned from pc.
        s   -- your string
    
    Keyword Arguments:
       loc  -- location. For now, 'lower right' and 'upper right'
               is implemented, or an explicit tuple of position of
               x and y.
       fig  -- use figtext instead of axes.text
       **kw -- keywords for call to text.
    '''
    import matplotlib.pyplot as plt;
    if   loc == 'lower right':
        x, y = 0.01, 0.02;
    elif loc == 'upper right':
        x, y = 0.01, 0.92;
    elif type(loc) == tuple:
        x,y = loc[:2]
    else:
        raise ValueError("unknown loc \"{}\"".format(loc));
    if fig:
        plt.figtext(x, y, s,**kw);
    else:
        ret['axes'].text(
            x, y, s,
            transform=ret['axes'].transAxes,
            **kw);
    pass

def highlight(ret, val,
              q=None,
              p=None,
              color='white', alpha=0.15,
              erase=False, cbar_lines=True,
              style='solid',
              empty_safe=True):
    '''
    Highlight a pc. Essentially a wrapper of plt.contour
    
    Arguments:
      ret   -- dict returned from pc.
      val   -- value to highlight.
      q     -- quantity to highlight. If None, highlight ret's quantity.
      p     -- grid. If None, use ret's quantity's dimensions.
    
    Keyword Arguments:
      color        -- color of highlight
      alpha        -- alpha of highlight
      cbar_lines   -- add lines to colorbar
      erase        -- erases the highlights. Defaults to 
                      false (opposite of matplotlib!)
    
    Returns:
      ret but with stuff that plt.contour adds.
    '''
    ax = ret['axes'];
    if p is None: p = ret['x'],ret['y'];
    x,y = p;
    if q is None:
        q = ret['q'];
    if empty_safe:
        mx,mn = q.max(),q.min();
        if val > mx or val < mn: return ret;
    if test(ret,'flip') or test(ret,'rotate'):
        x,y=y,x;
    #elif q is not ret['q'] and test(ret,'flip'):
    if not test(ret, 'cts'):
        ret['cts'] = [];
    ct = ax.contour(x,y,q, [val],
                    colors=[color], alpha = alpha,
                    linestyles=style,);
    ret['cts'].append(ct);
    if cbar_lines and ret['cbar'] is not None and q is ret['q']:
        ret['cbar'].add_lines(ct,erase=erase);
    return ret;

def quiv(ret, uv,
         C=None,
         p=None,
         color='black',
         qscale=1.0,
         maxscale=None,
         norm=None,
         copy=False,
         skip=None,
         scale=1.0,
         scale_units='xy',
         width=None,
         **kw):
    '''
    Place a quiver on a pc. Essentially a wrapper of plt.quiver
    
    Arguments:
      ret   -- dict returned from pc.
      uv    -- pair to quiver
      q     -- quantity to highlight. If None, highlight ret's quantity.
      p     -- grid. If None, use ret's quantity's dimensions.
    
    Keyword Arguments:
      C            -- quantity that maps to a colormap. Super
                      cedes `color` (see matplotlib.pyplot.quiver)
      color        -- color of arrows
      qscale       -- not to be confused with scale. My only flavor of
                      quiver scaling.
      maxscale     -- set the maximum length of vectors.
      norm         -- provide norm in order to keep from calculating it.
      copy         -- copy u and v to avoid modifying the vector.
      skip         -- two tuple of skips along each dimension in p's
                      units.
      scale        -- argument passed to Axes.quiver. Use `qscale`
                      instead [default: 1.0]
      scale_units  -- argument passed to Axes.quiver. [default 'xy']
      width        -- argument passed to Axes.quiver.

    
    Returns:
      ret but with stuff that plt.contour adds.
    '''
    ax = ret['axes'];
    if p is None: p = ret['x'],ret['y'];
    x,y = p;
    u,v = uv;
    if copy:
        u = np.copy(u);
        v = np.copy(v);
    if maxscale is not None:
        if norm is None: norm = np.sqrt(u**2+v**2);
        if maxscale == 0.0: raise ValueError(
                "dont quiver a zero maxscale");
        sel = norm > maxscale;
        s = maxscale/norm[sel];
        u[sel]*= s;
        v[sel]*= s;
    dx = x[1]-x[0];
    dy = y[1]-y[0];
    if skip:
        xskip,yskip = skip;
        xskip = np.round(xskip/dx).astype(int);
        yskip = np.round(yskip/dy).astype(int);
    else:
        xskip,yskip=1,1;
    xp = x[::xskip];
    yp = y[::yskip];
    
    u  = u[::yskip, ::xskip]/qscale*dx;
    v  = v[::yskip, ::xskip]/qscale*dy;
    #matplotlib developers should get covid19 and die
    args = [xp,yp,u,v];
    if C is not None: args+=[C];
    qv = ax.quiver(*args,
                   scale=scale,
                   color=color,
                   scale_units='xy',
                   **kw);
    if not test(ret, 'quivs'):
        ret['quivs'] = [];
    ret['quivs'].append(qv);
    return ret;
    
trajdefaults = dict(
    alpha = None,
    coords= ['x','y'],
    color = 'black',
    no_resize=False,
    cmap=None,
    color_quantity=None,
    marker='o',
    size=1,
    lw=0.1,
    scale =[1.0,1.0],
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
      coords    -- coordinates to plot as1l2 list of field names
      no_resize -- avoid resizing the axekms which happens if the
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
      scale     -- scale the coordinates.
      simple    -- simple scatter.
      flip      -- flip instead of rotate. Mimics behavior before version
                   0.0.12.
    Returns:
      None.
    '''
    import matplotlib.pyplot as plt;
    getkw=mk_getkw(kw, trajdefaults);
    xl,yl = getkw("coords");
    xs,ys = getkw("scale");
    if test(kw,'flip') or test(ret,'flip'):
        xl,yl = yl,xl; # yes, unneeded, but clearer.
        xs,ys = ys,xs;
    else:
        xl,yl = yl,xl;
        xs,ys =-ys,xs;
    if not test(kw, "no_resize"):
        xlim, ylim = ret['axes'].get_xlim(), ret['axes'].get_ylim();
    alpha = getkw('alpha');
    af = alpha;
    if alpha is None:
        af = lambda itr: None;
    elif type(alpha) == float:
        af = lambda itr: alpha;
    def nonnan(x):
        if x is not None:
            x = x.ravel();
            return x[np.isfinite(x)];
    if test(kw,'color_quantity'):
        cf = getkw('color_quantity');
        if type(cf) == str:
            cf = lambda itr: itr[cf];
        def _plotit(itr):
            x=nonnan(itr[xl])*xs;
            y=nonnan(itr[yl])*ys;
            x, y=x[s], y[s]; 
            ret['axes'].scatter(
                x, y,
                c=nonnan(cf(itr)),
                marker=getkw('marker'),
                lw=getkw('lw'),
                s=getkw('size'),
                #this is disabled pending further study
                #alpha=nonnan(af(itr)),
                cmap=getkw('cmap'));
        plotit = _plotit;
    else:
        #this must be plot for just alpha
        def _plotit(itr):
            x=nonnan(itr[xl])*xs;
            y=nonnan(itr[yl])*ys;
            ret['axes'].plot(
                x, y,
                lw=getkw('lw'),
                alpha=af(itr),
                c=getkw('color'),);
        plotit = _plotit;
    if test(kw, 'simple'):
        plotit(trajs);
    else:
        for itr in trajs:
            if np.any(np.isnan(itr[xl])):
                print("skipping nan");
                continue;
            plotit(itr);
    if not test(kw, "no_resize"):
        ret['axes'].set_xlim(xlim);
        ret['axes'].set_ylim(ylim);
    
