#!/usr/bin/env python
'''
Show an angular/energy/charge plot movie.

Usage:
  angularmov.py [options] <input> [<output>]

Options:
  --angle-bins=BINS -a BINS   Set the number of angle bins.  [default: 180]
  --energy-bins=BINS -r BINS  Set the number of energy bins. [default: 40]
  --ltitle=TITLE -t TITLE     Set the title.
  --rtitle=TITLE -T TITLE     Set the right title.
  --clabel=CLABEL -c CLABEL   Set colorbar label. [default: $p C$]
  --no-cbar                   Turn off the colorbar.
  --keV -k                    Use KeV. Included for backwards compatibility.
  --energy-units=UNITS        Set the energy units. Options are MeV, KeV, and eV [default: auto]
  --max-e=MAXE -e MAXE        Set the maximum energy value in the supplied units.
                              Use auto to auto scale by the max energy. [default: auto]
  --e-step=ESTEP              Set the step of grid lines for Energy. With max_e='auto', generate
                              this automatically.
  --angle-range=A -R A        Only plot this range of angles, does not change binning which is still
                              -pi to pi. Pass the values as fractions of pi. Example, -A (-0.5,0).
  --high-res -H               Output a high resolution plot.
  --max-q=MAXQ -q MAXQ        Set the maximum for the charge (pcolormesh's vmax value).
  --min-q=MINQ                Set a minimum charge.
  --normalize -n              Normalize the histogram to MeV^-1 rad^-1 .
  --factor=F -f F             Multiply histogram by F. [default: 1.0]
  --polar -p                  Plot polar angles for 3D data, letting the east direction be forward.
  --oap=ANGLE -o ANGLE        Set the width angle of the OAP. [default: 50.47]
  --massE=E                   Set the mass of the particles for efficiency in eV. [default: 0.511e6]
  --log10 -l                  Plot a logarithmic pcolor instead of a linear one.
  --cmap=CMAP                 Use the following cmap [default: viridis].
  --e-direction=ANGLE         The angle for the radial labels.
  --e-units=UNIT              The units for the radial labels.
  --agg -A                    Use the agg backend.
  --lsp -L                    Search for the lsp file in the current directory.
  --efficiency=E -E           Calculate the efficiency and display it. If you're lucky enough
                              to format your .lsp files the way I do, you can just set this to 
                              "wilks" and send the -L flag. Otherwise, you can either specify
                              an energy cut off, or pass a tuple of the form
                              (I, w, l, T, dim, [ecut]), where I is the intensity in W/cm^2,
                              w is the gaussian radius, l is the wavelength, T is the FWHM, and
                              dim is the number of dimensions. All units are SI unless otherwise
                              stated. ecut is an optional energy cut. If it is omitted, I'll
                              use the wilks scaling of the laser parameters given me.
  --interval=I                Set animate interval. [default: 100]1
  --timestep=S -s S           Set timestep in ns. [default: 1e-6]
  --initial-time=T -i T       Set initial timestep in ns. [default: 0]
  --minus-time=T -m T         Subtract this time. [default: 0]
  --timestamp-pos=POS         Set the timestamp position as a tuple [default: (0.02,0.05)];

'''
from docopt import docopt;
import numpy as np;
import matplotlib.pyplot as plt;
import matplotlib.animation as anim;
from lspplot.angular import _prep2,angular
from pys import test,parse_ftuple;
opts = docopt(__doc__,help=True);
d,kw = _prep2(opts);

tstep = float(opts['--timestep']);
ti    = float(opts['--initial-time']);
mt    = float(opts['--minus-time']);
tx,ty  = parse_ftuple(opts['--timestamp-pos']);

interval=float(opts['--interval']);
#process by times.
good = np.argsort(d['t'])
d = d[good];
t = d['t'];
tbins = np.arange(ti,t[-1]+tstep,tstep);
#fucking c like loop shit mother fucker.
i=0;
Is=[];
for j,ct in enumerate(t):
    if ct > tbins[i]:
        Is.append(j);
        i+=1;
#do first
surf,ax,fig,bins,data = angular(d,**kw);
kw['fig'] = fig;
rmax=ax.get_rmax()
t=fig.text(tx,ty,'t = {:4.1f} fs'.format(tbins[0]*1e6),
           fontdict={'fontsize':22});
if not test(kw, 'phi'):
    kw['phi']='phi'

def animate(ii):
    j,i = ii;
    #plt.clf();
    #_,ax,_,_ = angular(d[:i],**kw);
    #ax.set_rmax(rmax);
    S,_,_ = np.histogram2d(
        data[0][:i],
        data[1][:i],
        bins=bins,
        weights=data[2][:i]);
    surf.set_array(S[::,:-1].ravel());
    #t=fig.text(tx,ty,'t = {:3.2f}e-4 ns'.format((tbins[j]-mt)*1e4),
    #       fontdict={'fontsize':22});
    t.set_text('t = {:4.1f} fs'.format((tbins[j]-mt)*1e6));
    return surf;

a=anim.FuncAnimation(fig, animate,
                     list(enumerate(Is[1:])),
                     interval=interval);
if opts['<output>']:
    a.save(opts['<output>'],fps=15);
else:
    plt.show();
pass;
    
