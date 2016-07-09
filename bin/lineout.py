#!/usr/bin/env python2
'''
Usage:
    lineout.py [options] <i> [<output>]

Options:
    --xmin=MIN       Min [default: None]
    --xmax=MAX       Max [default: None]
    --width=W -w W   Width [default: 5]
'''

import numpy as np
from lspreader.flds import read_indexed, restrict
from docopt import docopt;

opts = docopt(__doc__,help=True);
svar = ["RhoN{}".format(i) for i in range(2,12)]
i = int(opts['<i>'])
xi=  (eval(opts['--xmin']), eval(opts['--xmax']))
print(xi)
d = read_indexed(i,
    sclr=svar,
    gzip=True,dir='.');
ii = list(range(2,10)) + [11]
qs = list(range(1,8)) + [1] 
ni = sum([q*d['RhoN{}'.format(i)] for q,i in zip(qs,ii)])
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt;

ne = d['RhoN10']
yi =  ne.shape[1]/2;
w = int(opts['--width'])/2;
ne = ne[xi[0]:xi[1],yi-w:yi+w]
ni = ni[xi[0]:xi[1],yi-w:yi+w]
ne = np.average(ne,axis=1)
ni = np.average(ni,axis=1)
x = d['x'][xi[0]:xi[1],yi]*1e4
plt.semilogy(x,ne,label='n_e');
plt.semilogy(x,ni,label='n_i');
plt.legend()
plt.ylabel("number/cc");
plt.xlabel("microns");
plt.title("Density");
name = opts['<output>'] if opts['<output>'] else 'lineout.png'
plt.savefig(name)
