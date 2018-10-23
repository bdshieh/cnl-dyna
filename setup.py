## interaction / setup.py
'''
'''
from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize

import numpy as np
import os

# set language argument based on OS
if os.name == 'nt':
    _LANGUAGE = 'c++' # for MSVC++
else:
    _LANGUAGE = 'c' # for gcc

_LANGUAGE = 'c'

ext_modules = [
    Extension(name='pyh2lib',
              sources=['cmut_nonlinear_sim/core/pyh2lib.pyx'],
              include_dirs=['include'],
              libraries=['math', 'libH2LIB', 'openblas'],
              library_dirs=['lib'],
              language=_LANGUAGE
    )
]


setup(
    name='cmut-nonlinear-sim',
    version='0.1',
    ext_modules=cythonize(ext_modules),
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy'
    ],
    setup_requires=[
        'setuptools>=18.0', 
        'cython>=0.25']
)

