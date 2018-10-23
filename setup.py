## interaction / setup.py
'''
'''
from setuptools import setup, find_packages
from setuptools.extension import Extension

import numpy as np
import os

# set language argument based on OS
if os.name == 'nt':
    _LANGUAGE = 'c++' # for MSVC++
else:
    _LANGUAGE = 'c' # for gcc

ext_modules = [
    Extension(name='cmut-nonlinear-sim.core.h2lib',
              sources=['cmut-nonlinear-sim/core/h2lib.pyx'],
              include_dirs=['include'],
              libraries=['lmath', 'libH2LIB', 'openblas'],
              library_dirs=['lib'],
              language=_LANGUAGE
    )
]


setup(
    name='cmut-nonlinear-sim',
    version='0.1',
    ext_modules=ext_modules,
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy'
    ],
    setup_requires = ['setuptools>=18.0', 'cython>0.25']
)

