from setuptools import setup

setup(name='lspplot',
      version='0.0.12r1',
      description='lsp plots',
      url='http://github.com/noobermin/lspplot',
      author='noobermin',
      author_email='ngirmang.1@osu.com',
      license='MIT',
      install_requires=[
          'pys>=0.0.9',
          'lspreader>=0.1.5',
          'matplotlib>=1.5.3',
          'numpy>=1.10.4',
          'scipy>=0.18.1'],
      packages=['lspplot'],
      zip_safe=False);
