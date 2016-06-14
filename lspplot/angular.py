'''
Functions for creating my angular plots.
'''
import numpy as np;
import matplotlib;
import matplotlib.pyplot as plt;
import matplotlib.patheffects as pe;
from matplotlib import colors;
from pys import test,mk_getkw,parse_ftuple,sd;
from cmaps import pastel_clear,plasma_clear,viridis_clear,magma_clear_r;
import re;
from physics import laserE;

def _getlsp(path=None):
    if not path:
        import os;
        path = [ f for f in 
                 os.listdir(".")
                 if re.search(".*\.lsp$", f)][0];
    with open(path,'r') as f:
        lines = f.read()
    getrx = lambda rx:float(re.search(rx,lines,flags=re.MULTILINE).group(1));
    I = getrx("intensity=(.*) W/cm\^2 *$");
    #FWHM so divide by 2
    T =getrx("FWHM *\n *independent_variable_multiplier *(.*)$")*1e-9/2.0;    
    w =getrx(
        "\\lambda *spotsize *\n *coefficients [0-9e\-\.]+ *(.*) *end$")*1e-2;
    xcells = getrx("x-cells *([0-9]+)");
    ycells = getrx("y-cells *([0-9]+)");
    zcells = getrx("z-cells *([0-9]+)");
    dims=0;
    if xcells > 0:dims+=1;
    if ycells > 0:dims+=1;
    if zcells > 0:dims+=1;
    dim="{}D".format(dims);
    return I,w,T,dim;

def totalKE(d, ecut=0, anglecut=None,return_bools=False):
    '''
    Get total energy across a pext plane.

    Parameters and Keywords
    -----------------------
    d            -- array
    ecut         -- energy cut in eV
    anglecut     -- None or a tuple of (angle, dimension) where dimension
                    is "2D" or "3D". If None or falsey, don't cut on angle.
    return_bools -- return the boolean array that selects uncut particles.

    Returns total energy.
    '''
    good = d['KE'] > ecut;
    if anglecut:
        angle, dim = anglecut;
        if dim == '2D':
            good &= np.abs(d['phi']) > np.pi - angle/180*np.pi;
        elif dim == '3D':
            good &= np.cos(angle/180*np.pi) < -np.sin(d['theta'])*np.cos(d['phi']);
        else:
            raise ValueError("anglecut is not None, '2D' or '3D'");
    KE = (np.abs(d['q']*1e-6)*d['KE'])[good].sum();
    if return_bools:
        return KE,good;
    else:
        return KE;


defaults = {
    'tlabels': ['Forward\n0$^{\circ}$',
               '45$^{\circ}$',
               'Up\n90$^{\circ}$',
               '135$^{\circ}$',
               'Backwards\n180$^{\circ}$',
               '215$^{\circ}$',
               'Down\n270$^{\circ}$',
               '315$^{\circ}$'],
    'labels': ['Forward\n0$^{\circ}$',
               '45$^{\circ}$',
               'Left\n90$^{\circ}$',
               '135$^{\circ}$',
               'Backwards\n180$^{\circ}$',
               '215$^{\circ}$',
               'Right\n270$^{\circ}$',
               '315$^{\circ}$'],
    'angle_bins': 180,
    'energy_bins': 40,
    'max_e': 4.0,
    'max_e_kev': 1000,
    'e_step' : 1.0,
    'e_step_kev': 250,
    'max_q': None,
    'min_q': None,
    'cmap': pastel_clear,
    'clabel': '$pC$',
    'log_q' : None,
    'norm_units': ' rad$^{-1}$ MeV$^{-1}$',
};

rgrid_defaults = dict(
    unit='MeV',
    unit_kw='keV',
    angle=45,
    size=10.5,
    color='gray');

def getkw(kw,label):
    return kw[label] if test(kw,label) else defaults[label];
def getkw_kev(kw,label,kev):
    if test(kw,label):
        return kw[label];
    if kev: label+='_kev';
    return defaults[label];
def angular(d, phi=None, e=None,
            colorbar=True,**kw):
    '''
    Make the angular plot.
    Call form 1:

    angular(s, phi, e, kw...)

    Arguments:
      s   -- the charges.
      phi -- the angles of ejection.
      e   -- energies of each charge.

    Call form 2

    angular(d, kw...)
    Arguments:
      d   -- pext data, a structured array from the lspreader.

    Keyword Arugments:
      phi         -- If a string, read this array of recs as
                     the angles. If an array, these are the angles.
      e           -- If not None, use these as energies over d['KE']
      F           -- Multiply charges by a factor.
      max_e       -- Maximum energy, if 'auto',
                     bin automatically.
      max_q       -- Maximum charge.
      angle_bins  -- Set the number of angle bins.
      energy_bins -- Set the number of energy bins.
      clabel      -- Set the colorbar label.
      colorbar    -- If true, plot the colorbar.
      e_step      -- Set the steps of the radius contours.
      labels      -- Set the angular labels. If not a list, if
                     'default', use default. If 'tdefault', use
                     default for theta. (See defaults dict);
      keV         -- Use keV instead of MeV.
      fig         -- If set, use this figure, Otherwise,
                     make a new figure.
      ax          -- If set, use this axis. Otherwise,
                     make a new axis.
      ltitle      -- Make a plot on the top left.
      rtitle      -- Make a plot on the top right.
      log_q       -- log10 the charges.
      cmap        -- use the colormap cmap.
      rgridopts   -- pass a dictionary that sets details for the
                     rgrid labels.
      oap         -- Plot this half angle if not None.
      normalize   -- Subdivide charges by the bin weights.
      efficiency  -- calculate and display the conversion efficiency in
                     the oap angle. A dict of options passed to totalKE
                     and laserE (I only, not E_0). See help for totalKE
                     and laserE.
    '''
    #reckon the call form
    kev = test(kw, 'keV');
    if type(d) == np.ndarray and len(d.dtype) > 0:
        structd = True;
        e = np.copy(d['KE']);
        if not phi: phi = 'phi';
        phi = np.copy(d[phi]);
        s = np.abs(d['q'])*1e6;
    else:
        structd = False;
        if phi is None or e is None:
            raise ValueError("Either d isn't a recarray or phi and s were not passed.")
        s = d;
    if kev:
        e/=1e3;
    else:
        e/=1e6;
    if test(kw, 'F'): s*=kw['F'];
    phi_spacing = getkw(kw,'angle_bins');
    E_spacing =   getkw(kw,'energy_bins');
    maxE  = getkw_kev(kw,'max_e',kev);
    Estep = getkw_kev(kw,'e_step',kev);
    if maxE == 'max':
        maxE = np.max(e);
    elif maxE == 'round' or maxE == 'auto':
        mxe = np.max(e);
        tenpow = np.floor(np.log10(mxe))
        maxE = 10**tenpow * (int(mxe/(10**tenpow))+1)
        Estep = maxE/4.0;
        if 4.0 > maxE > 2.0:
            Estep = 1.0;
        elif 2.0 >= maxE > 1.0:
            Estep = 0.5;
        elif Estep < 1.0:
            Estep = int(Estep/0.25)*0.25;
        else:
            #round up Estep to either a multiple of five or two
            nearest2 = int(Estep / 2.0) * 2;
            nearest5 = int(Estep / 5.0) * 5;
            if abs(nearest2 - Estep) < abs(nearest5 - Estep):
                Estep = nearest2;
            else:
                Estep = nearest5;
    maxQ  = getkw(kw,'max_q');
    minQ  = getkw(kw,'min_q');
    if test(kw,"normalize"):
        s /= maxE/E_spacing*2*np.pi/phi_spacing;
    clabel = getkw(kw,'clabel');
    cmap = getkw(kw, 'cmap');
    phi_bins = np.linspace(-np.pi,np.pi,phi_spacing+1);
    E_bins   = np.linspace(0, maxE, E_spacing+1);
            
    PHI,E = np.mgrid[ -np.pi : np.pi : phi_spacing*1j,
                      0 : maxE : E_spacing*1j];
    S,_,_ = np.histogram2d(phi,e,bins=(phi_bins,E_bins),weights=s);
    if test(kw,'fig'):
        fig = kw['fig']
    else:
        fig = plt.figure(1,facecolor=(1,1,1));
    if test(kw,'ax'):
        ax= kw['ax']
    else:
        ax= plt.subplot(projection='polar',axisbg='white');
    norm = matplotlib.colors.LogNorm() if test(kw,'log_q') else None;
    
    surf=plt.pcolormesh(PHI,E,S,norm=norm, cmap=cmap,vmin=minQ,vmax=maxQ);
    if colorbar:
        c=fig.colorbar(surf,pad=0.1);
        c.set_label(clabel);
    #making radial guides. rgrids only works for plt.polar calls
    #making rgrid
    if test(kw, 'rgridopts'):
        ropts = kw['rgridopts'];
    else:
        ropts = dict();
        
    getrkw = mk_getkw(ropts,rgrid_defaults);
    if test(ropts, 'unit'):
        runit = ropts['unit'];
    else:
        runit = 'keV' if kev else 'MeV';
    rangle=getrkw('angle');
    rsize =getrkw('size');
    gridc =getrkw('color');
    if test(ropts, 'invert'):
        c1,c2 = "w","black";
    else:
        c1,c2 = "black","w";
    full_phi = np.linspace(0.0,2*np.pi,100);
    for i in np.arange(0.0,maxE,Estep)[1:]:
        plt.plot(full_phi,np.ones(full_phi.shape)*i,
                 c=gridc, alpha=0.9,
                 lw=1, ls='--');
    ax.set_theta_zero_location('N');
    ax.patch.set_alpha(0.0);
    ax.set_axis_bgcolor('red');
    rlabel_str = '{} ' + runit;
    rlabels    = np.arange(0.0,maxE,Estep)[1:];
    #text outlines.
    _,ts=plt.rgrids(rlabels,
                    labels=map(rlabel_str.format,rlabels),
                    angle=rangle);
    for t in ts:
        t.set_path_effects([
            pe.Stroke(linewidth=1.5, foreground=c2),
            pe.Normal()
        ]);
        t.set_size(rsize);
        t.set_color(c1);
    if test(kw,'oap'):
        oap = kw['oap']/2 * np.pi/180;
        maxt = oap+np.pi; mint = np.pi-oap;
        maxr  = maxE*.99;
        if test(kw, 'efficiency') and structd:
            defeff = dict(
            I=3e18,w=None,T=None,
                ecut=0, anglecut=None);
            effd=sd(defeff,**kw['efficiency']);
            minr = effd['ecut'] * (1e-3 if kev else 1e-6);
            LE=laserE(I=effd['I'],w=effd['w'],T=effd['T']);
            KE,good=totalKE(d,ecut = effd['ecut'],anglecut=(oap/np.pi*180,effd['dim']),
                       return_bools=True)
            totalq = np.abs(d['q'][good]).sum()*1e6;
            dim = "3D" if not effd['anglecut'] else effd['anglecut'][1];
            def texs(f,l=2):
                tenpow=int(np.floor(np.log10(f)));
                nfmt = "{{:0.{}f}}".format(l) + "\\cdot 10^{{{}}}"
                return nfmt.format(f/10**tenpow,tenpow);
            fig.text(
                0.01,0.04,
                "Efficiency:\n$\\frac{{{}J}}{{{}J}}$=${}$".format(
                    texs(KE,l=1),texs(LE,l=1),texs(KE/LE)),
                fontdict=dict(fontsize=20));
            fig.text(
                0.65,0.05,
                "$Q_{{tot}}={} ${}".format(
                    texs(totalq), "pC" if dim == "3D" else "pC/cm"),
                fontdict=dict(fontsize=20));
            fig.text(
                0.6,0.92,
                "I = ${}$ W/cm$^2$".format(
                    texs(effd['I'],l=1)),
                fontdict=dict(fontsize=20));
        else:
            minr = 120 if kev else 0.12;
        ths=np.linspace(mint, maxt, 20);
        rs =np.linspace(minr, maxr, 20);
        mkline = lambda a,b: plt.plot(a,b,c=(0.2,0.2,0.2),ls='-',alpha=0.5);
        mkline(ths, np.ones(ths.shape)*minr)
        mkline(mint*np.ones(ths.shape), rs);
        mkline(maxt*np.ones(ths.shape), rs);
    if test(kw,'labels'):
        if kw['labels'] == 'default':
            labels = defaults['labels'];
        elif kw['labels'] == 'tdefault':
            labels = defaults['tlabels'];
        else:
            labels= kw['labels'];
        ax.set_xticks(np.pi/180*np.linspace(0,360,len(labels),endpoint=False));
        ax.set_xticklabels(labels);
    if test(kw,'ltitle'):
        if len(kw['ltitle']) <= 4:
            ax.set_title(kw['ltitle'],loc='left',fontdict={'fontsize':24});
        else:
            ax.text(np.pi/4+0.145,maxE+Estep*2.4,kw['ltitle'],fontdict={'fontsize':24});
    if test(kw,'rtitle'):
        if '\n' in kw['rtitle']:
            fig.text(0.60,0.875,kw['rtitle'],fontdict={'fontsize':22});
        else:
            plt.title(kw['rtitle'],loc='right',fontdict={'fontsize':22});
    return (surf, ax, fig, (phi_bins, E_bins));




def angular_load(
        fname,polar=False):
    '''
    load the pext data and normalize

    parameters:
    -----------

    fname       -- name of file or data
    F           -- Factor to scale by, None doesn't scale.
    normalize   -- if None, don't normalize. Otherwise, pass a dict with
                   {'angle_bins': ,'energy_bins': , 'max_e': }
                   with the obvious meanings. Normalize with max_phi as phi.
    polar       -- if polar, use phi_n over phi in the file/data.
    '''
    d = np.load(fname, allow_pickle=True);
    e = d['KE'];
    phi = d['phi_n'] if polar else d['phi'];
    s   = np.abs(d['q']*1e6);
    return s,phi,e,d;

#this is garbage. Done for compatibility/code share with angularmov.py



def _prep(opts):
    kw=_prepkw(opts);
    #this deals with pre-processing.
    s,phi,e,d = angular_load(
        opts['<input>'],
        polar=opts['--polar'])
    return s,phi,e,kw,d;
def _prep2(opts):
    kw=_prepkw(opts);
    #this deals with pre-processing.
    return np.load(opts['<input>']) , kw;

def _prepkw(opts):
    '''Prep from options'''
    inname = opts['<input>'];
    kev = opts['--keV'];
    def getdef_kev(label):
        
        if kev:
            return defaults[label+'_kev'];
        else:
            return defaults[label];
    kw = {
        'angle_bins' : float(opts['--angle-bins']),
        'energy_bins': float(opts['--energy-bins']),
        'max_q': float(opts['--max-q']) if opts['--max-q'] else None,
        'min_q': float(opts['--min-q']) if opts['--min-q'] else None,
        'keV': kev,
        'clabel' : opts['--clabel'],
        'colorbar' : not opts['--no-cbar'],
        'e_step' : float(opts['--e-step']) if opts['--e-step'] else None,
        'labels': 'tdefault' if opts['--polar'] else 'default',
        'rtitle':opts['--rtitle'],
        'ltitle':opts['--ltitle'],
        'oap': float(opts['--oap']) if opts['--oap'] != 'none' else None,
        'log_q': opts['--log10'],
        'normalize':opts['--normalize'],
        'F':float(opts['--factor']),
    };
    if opts['--max-e']:
        try:
            kw['max_e']=float(opts['--max-e']);
        except ValueError:
            kw['max_e']=opts['--max-e'];
    else:
        kw['max_e'] = getdef_kev('max_e');
    cmap = _str2cmap(opts['--cmap']);
    if not cmap:
        cmap = opts['--cmap'];
    kw['cmap'] = cmap;
    kw['rgridopts'] = {};
    if opts['--e-direction']:
        kw['rgridopts'].update({'angle':opts['--e-direction']});
    if opts['--e-units']:
        kw['rgridopts'].update({'unit':opts['--e-units']});
    if opts['--normalize']:
        kw['clabel'] += defaults['norm_units'];
    if opts['--lsp'] and opts['--efficiency']:
        I,w,T,dim = _getlsp();
        ecut = float(opts['--efficiency']);
        if kev:
            ecut*=1e3;
        else:
            ecut*=1e6;
        kw['efficiency'] = dict(
            I=I,w=w,T=T,dim=dim,
            ecut=ecut);
    return kw;




def _str2cmap(i):    
    if i == 'viridis_clear':
        return viridis_clear;
    elif i == 'plasma_clear':
        return plasma_clear;
    elif i == 'magma_clear_r':
        return magma_clear_r;
    elif i == 'pastel_clear':
        return pastel_clear;
    elif i == 'pastel':
        return pastel;
    pass;
