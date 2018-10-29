## interaction / setup.py
'''
'''
from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize

import numpy as np
# import os


ext_modules = [
    Extension(name='cmut_nonlinear_sim.core.libh2_classes',
              sources=['cmut_nonlinear_sim/core/libh2_classes.pyx'],
              include_dirs=['include'],
              libraries=['h2', 'openblas'],
              library_dirs=['lib'],
              language='c',
              extra_compile_args=['-fPIC']
    ),
    Extension(name='cmut_nonlinear_sim.core.libh2_functions',
              sources=['cmut_nonlinear_sim/core/libh2_functions.pyx'],
              include_dirs=['include'],
              libraries=['h2', 'openblas'],
              library_dirs=['lib'],
              language='c',
              extra_compile_args=['-fPIC']
    )
]


setup(
    name='cmut-nonlinear-sim',
    version='0.1',
    ext_modules=cythonize(ext_modules),
    packages=find_packages(),
    install_requires=[
    ],
    setup_requires=[
        'setuptools>=18.0', 
        'cython>=0.25']
)

