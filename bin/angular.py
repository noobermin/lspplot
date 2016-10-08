#!/usr/bin/env python2
'''
Show an angular/energy/charge plot.

Usage:
  angular.py [options] <input> [<output>]

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
  --efficiency=E              Calculate the efficiency and display it, pass a tuple
                              of the energy cut.
'''

from docopt import docopt;
import numpy as np;
from lspplot.angular import angular,_prep2;
import matplotlib.pyplot as plt

opts=docopt(__doc__,help=True);
d,kw = _prep2(opts);
if opts['<output>'] and opts['--agg']:
    plt.switch_backend('agg');
angular(d,**kw);
if opts['<output>']:
    if opts['--high-res']:
        plt.savefig(opts['<output>'],dpi=1000);
    else:
        plt.savefig(opts['<output>']);
else:
    plt.show();
pass;
