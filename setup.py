from setuptools import setup

setup(name='lspplot',
      version='0.1.4',
      description='lsp plots',
      url='http://github.com/noobermin/lspplot',
      author='noobermin',
      author_email='ngirmang.1@osu.com',
      license='MIT',
      install_requires=[
          'pys>=0.0.12r1',
          'lspreader>=0.1.7',
          'matplotlib>=2.0.0',
          'numpy>=1.10.4',
          'scipy>=0.18.1'],
      packages=['lspplot'],
      scripts=[
          'bin/EM.py',
          'bin/sclrq.py',
          'bin/ion.py',
          'bin/rho.py',
          'bin/angular.py',
          'bin/angularmov.py',
          'bin/energy_cons.py',
      ],
      zip_safe=False);
