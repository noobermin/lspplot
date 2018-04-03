'''
Functions for creating my angular plots.
'''
import numpy as np;
import matplotlib;
import matplotlib.pyplot as plt;
import matplotlib.patheffects as pe;
from lspreader.dotlsp import getdim;
from matplotlib import colors;
from pys import test,mk_getkw,parse_ftuple,sd;
from lspplot.cmaps import pastel_clear,plasma_clear,viridis_clear,magma_clear_r;
import re;
from lspplot.physics import laserE, a0;

def _getlsp(path=None):
    if not path:
        import os;
        path = [ f for f in 
                 os.listdir(".")
                 if re.search(".*\.lsp$", f)][0];
    with open(path,'r') as f:
        lsp = f.read()
    getrx = lambda rx:float(re.search(rx,lsp,flags=re.MULTILINE).group(1));
    I = getrx("intensity=(.*) W/cm\^2 *$");
    #FWHM so divide by 2
    T =getrx("FWHM *\n *independent_variable_multiplier *(.*)$")*1e-9/2.0;    
    w =getrx(
        "\\lambda *spotsize *\n *coefficients [0-9e\-\.]+ *(.*) *end$")*1e-2;
    l =getrx(
        "\\lambda *spotsize *\n *coefficients ([0-9e\-\.]+) *.* *end$")*1e-2;
    dim="{}D".format(
        len(getdim(lsp)));
    return I,w,l,T,dim;

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


defaults = dict(
    tlabels=['Forward\n0$^{\circ}$',
             '45$^{\circ}$',
             'Up\n90$^{\circ}$',
             '135$^{\circ}$',
             'Backwards\n180$^{\circ}$',
             '215$^{\circ}$',
             'Down\n270$^{\circ}$',
             '315$^{\circ}$'],
    labels=['Forward\n0$^{\circ}$',
            '45$^{\circ}$',
            'Left\n90$^{\circ}$',
            '135$^{\circ}$',
            'Backwards\n180$^{\circ}$',
            '215$^{\circ}$',
            'Right\n270$^{\circ}$',
            '315$^{\circ}$'],
    energy_units='MeV',
    angle_bins=180,
    energy_bins=40,
    min_q = None,
    max_q = None,
    cmap  = pastel_clear,
    clabel= '$pC$',
    log_q = None,
    norm_units=' rad$^{-1}$ MeV$^{-1}$',
);
unit_defaults = dict(
    eV=dict(
        energy_scale=1,
        toMeV=1e6,
        max_e=800.0,  e_step=200.0),
    KeV=dict(
        energy_scale=1e3,
        toMeV=1e3,
        max_e=1000.0,e_step=250.0),
    MeV=dict(
        energy_scale=1e6,
        toMeV=1,
        max_e=4.0, e_step=1.0),
    GeV=dict(
        energy_scale=1e9,
        toMeV=1e-3,
        max_e=2.0, e_step=0.5),);



rgrid_defaults = dict(
    angle=45,
    size=10.5,
    color='gray');

def angular(d, phi=None, e=None,
            **kw):
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
      phi          -- If a string, read this array of recs as
                      the angles. If an array, these are the angles.
      e            -- If not None, use these as energies over d['KE']

      energy_units -- Set the energy units. Options are eV, KeV, MeV, GeV,
                      and auto. auto chooses the unit based on the max
                      energy.
      energy_scale -- Set the energy scale. Not required, but you can hack it
                      from the default due to energy_unit if you wish.
      toMeV        -- Scale to the MeV scale from energy scale. Not required,
                      but you can hack it from energy_unit if you wish.
      max_e        -- Maximum energy, if 'auto',
                      bin automatically.
      e_step       -- Set the steps of the radius contours.

      min_q        -- Minimum charge.
      max_q        -- Maximum charge.
      angle_range  -- Only bin and plot these angles. Does not affect angle
                      binning, bins are still over (-pi,pi)

      angle_bins   -- Set the number of angle bins.
      energy_bins  -- Set the number of energy bins.

      colorbar     -- If true, plot the colorbar.
      cmap         -- use the colormap cmap.
      clabel       -- Set the colorbar label.

      labels       -- Set the angular labels. If not a list, if
                      'default', use default. If 'tdefault', use
                      default for theta. (See defaults dict);

      normalize    -- Subdivide charges by the bin weights.

      fig          -- If set, use this figure, Otherwise,
                      make a new figure.
      ax           -- If set, use this axis. Otherwise,
                      make a new axis.
      ltitle       -- Make a plot on the top left.
      rtitle       -- Make a plot on the top right.
      log_q        -- log10 the charges.
      rgridopts    -- pass a dictionary that sets details for the
                      rgrid labels. Options for this dict are:
            angle     -- angle of labels.
            size      -- text side.
            color     -- grid color.
            invert    -- invert the rgrid colors.

      oap          -- Plot this apex angle if not None as the oap 
                      collection angle.

      efficiency   -- calculate and display the conversion efficiency in
                      the oap angle. A dict of options passed to totalKE
                      and laserE (I only, not E_0). See help for totalKE
                      and laserE.
      F            -- Multiply charges by a factor.
      dict_return  -- Have the return value be a convenient dictionary
                      instead of god knows what it currently is (a
                      tuple of tuple of stuff).

    '''
    #reckon the call form
    getkw = mk_getkw(kw,defaults);
    if type(d) == np.ndarray and len(d.dtype) > 0:
        structd = True;
        e = np.copy(d['KE']);
        if not phi: phi = 'phi';
        phi = np.copy(d[phi]);
        s = np.abs(d['q'])*1e6;
    else:
        structd = False;
        if phi is None or e is None:
            raise ValueError(
                "Either phi and s were not passed. See help.");
        s = d;
    eunits = getkw('energy_units');
    if eunits == 'auto':
        pw = np.log10(np.max(e))
        if pw < 3:
            eunits = 'eV';
        elif pw <= 5:
            eunits = 'KeV';
        elif pw <= 9:
            eunits = 'MeV';
        else:
            eunits = 'GeV';
    if test(kw,'angle_range'):
        mnang,mxang = kw['angle_range'];
        if mxang > np.pi:
            good = np.logical_and(phi >= mnang, phi <= np.pi);
            good|= np.logical_and(phi >= -np.pi, phi <= -(mxang-np.pi));
        else:
            good = np.logical_and(phi >= mnang, phi <=  mxang);
        phi = phi[good];
        e   =   e[good];
        s   =   s[good];
        if structd:
            d = d[good];
    getunitkw = mk_getkw(kw,unit_defaults[eunits]);
    if test(kw, 'F'): s*=kw['F'];
    phi_spacing = getkw('angle_bins');
    E_spacing =   getkw('energy_bins');
    e /= getunitkw('energy_scale');
    maxE  = getunitkw('max_e');
    Estep = getunitkw('e_step');
    if maxE == 'max':
        maxE = np.max(e);
    elif maxE == 'round' or maxE == 'auto':
        mxe = np.max(e);
        tenpow = np.floor(np.log10(mxe))
        mantissa = np.floor(mxe/(10**tenpow));
        maxE = 10**tenpow * (int(mxe/(10**tenpow))+1)
        Estep = 10**tenpow;
        if mantissa > 6:
            Estep = 6*10**tenpow;
    if test(kw,'e_step'):
        Estep = kw['e_step'];
    maxQ  = getkw('max_q');
    minQ  = getkw('min_q');
    if test(kw,"normalize"):
        s /= maxE/E_spacing*2*np.pi/phi_spacing;
        s *= getunitkw('toMeV');
    clabel = getkw('clabel');
    cmap = getkw('cmap');
    phi_bins = np.linspace(-np.pi,np.pi,phi_spacing+1);
    E_bins   = np.linspace(0, maxE, E_spacing+1);
            
    PHI,E = np.mgrid[ -np.pi : np.pi : phi_spacing*1j,
                           0 :  maxE :   E_spacing*1j];
    
    S,_,_ = np.histogram2d(phi,e,bins=(phi_bins,E_bins),weights=s);
    if test(kw,'fig'):
        fig = kw['fig']
    else:
        fig = plt.figure(1,facecolor=(1,1,1));
    if test(kw,'ax'):
        ax = kw['ax']
    else:
        ax = plt.subplot(projection='polar',facecolor='white');
    norm = matplotlib.colors.LogNorm() if test(kw,'log_q') else None;
    ax.set_rmax(maxE);
    surf=plt.pcolormesh(
        PHI,E,S,norm=norm,cmap=cmap,
        vmin=minQ,vmax=maxQ);
    if test(kw,'colorbar'):
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
        runit = eunits;
    rangle=getrkw('angle');
    rsize =getrkw('size');
    gridc =getrkw('color');
    if test(ropts, 'invert'):
        c1,c2 = "w","black";
    else:
        c1,c2 = "black","w";
    full_phi = np.linspace(0.0,2*np.pi,100);
    rlabels  = np.arange(0.0,maxE,Estep)[1:];
    for i in rlabels:
        plt.plot(full_phi,np.ones(full_phi.shape)*i,
                 c=gridc, alpha=0.9,
                 lw=1, ls='--');
    ax.set_theta_zero_location('N');
    ax.patch.set_alpha(0.0);
    ax.set_facecolor('red');
    rlabel_str = '{} ' + runit;
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
                I=3e18,w=None,T=None,l=None,
                ecut=0, anglecut=None);
            effd=sd(defeff,**kw['efficiency']);
            if effd['ecut'] == 'wilks':
                effd['ecut'] = (
                    np.sqrt(1+a0(effd['I'],l=effd['l']*1e2)**2/2.0) - 1.0
                )*effd['massE'];
            dim = effd['dim'];
            LE=laserE(I=effd['I'],w=effd['w'],T=effd['T'],dim=dim);
            KE,good=totalKE(
                d, ecut=effd['ecut'], anglecut=(oap/np.pi*180,dim),
                return_bools=True)
            minr = effd['ecut'] / getunitkw('energy_scale');
            totalq = np.abs(d['q'][good]).sum()*1e6;
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
            minr = 0.12/getunitkw('toMeV')
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
    if test(kw, 'dict_return'):
        return dict(
            surf=surf,
            ax=ax,
            fig=fig,
            phi_bins=phi_bins,
            E_bins=E_bins,
            phi=phi,
            e=e,
            s=s,
            eunits=eunits,
        );
    else:
        return (surf, ax, fig, (phi_bins, E_bins), (phi,e,s));




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
        opts['<input>'], polar=opts['--polar'])
    return s,phi,e,kw,d;
def _prep2(opts):
    kw=_prepkw(opts);
    #this deals with pre-processing.
    return np.load(opts['<input>']) , kw;

def _prepkw(opts):
    '''Prep from options'''
    inname = opts['<input>'];
    eunit  = opts['--energy-units'];
    if opts['--keV']:
        eunit='KeV';
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
        'energy_units':eunit,
    };
    if opts['--angle-range']:
        kw['angle_range'] = [
            x*np.pi for x in parse_ftuple(opts['--angle-range'])];
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
    
    if opts['--efficiency']:
        massE=float(opts['--massE']);
        effs = opts['--efficiency'];
        if opts['--lsp']:
            I,w,l,T,dim = _getlsp();
            if effs == "wilks":
                ecut = 'wilks';
            else:
                ecut = float(effs);
        else:
            vs = parse_ftuple(effs, length=None);
            if len(vs) >= 5:
                I,w,l,T,dim = vs[:5];
                dim = "{}D".format(int(dim));
                if len(vs) >= 6:
                    ecut = vs[5];
                else:
                    ecut = 'wilks';
            else:
                raise ValueError(
                    "efficiency passed is incorrect. See --help.");
        kw['efficiency'] = dict(
            I=I,w=w,T=T,l=l,dim=dim,
            ecut=ecut, massE=massE);
    if opts['--polar']:
        kw['phi']='phi_n';
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
