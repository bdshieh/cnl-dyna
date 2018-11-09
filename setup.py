## interaction / setup.py
'''
'''
from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize

import numpy as np
# import os


ext_opts = {
    'libraries':['h2', 'openblas'],
    'library_dirs':['lib'],
    'language':'c',
    'extra_compile_args':['-fPIC', '-DUSE_COMPLEX','-Wno-strict-prototypes']
}

ext_modules = [
    Extension(
        name='cmut_nonlinear_sim.h2lib.basic_cy',
        sources=['cmut_nonlinear_sim/h2lib/basic_cy.pyx'],
        **ext_opts
    ),
    Extension(
        name='cmut_nonlinear_sim.h2lib.avector_cy',
        sources=['cmut_nonlinear_sim/h2lib/avector_cy.pyx'],
        **ext_opts
    ),
    Extension(
        name='cmut_nonlinear_sim.h2lib.amatrix_cy',
        sources=['cmut_nonlinear_sim/h2lib/amatrix_cy.pyx'],
        **ext_opts
    ),
    Extension(
        name='cmut_nonlinear_sim.h2lib.cluster_cy',
        sources=['cmut_nonlinear_sim/h2lib/cluster_cy.pyx'],
        **ext_opts
    ),
    Extension(
        name='cmut_nonlinear_sim.h2lib.block_cy',
        sources=['cmut_nonlinear_sim/h2lib/block_cy.pyx'],
        **ext_opts
    ),
    Extension(
        name='cmut_nonlinear_sim.h2lib.surface3d_cy',
        sources=['cmut_nonlinear_sim/h2lib/surface3d_cy.pyx'],
        **ext_opts
    ),
    Extension(
        name='cmut_nonlinear_sim.h2lib.macrosurface3d_cy',
        sources=['cmut_nonlinear_sim/h2lib/macrosurface3d_cy.pyx'],
        **ext_opts
    ),
    Extension(
        name='cmut_nonlinear_sim.h2lib.rkmatrix_cy',
        sources=['cmut_nonlinear_sim/h2lib/rkmatrix_cy.pyx'],
        **ext_opts
    ),
    Extension(
        name='cmut_nonlinear_sim.h2lib.hmatrix_cy',
        sources=['cmut_nonlinear_sim/h2lib/hmatrix_cy.pyx'],
        **ext_opts
    ),
    Extension(
        name='cmut_nonlinear_sim.h2lib.bem3d_cy',
        sources=['cmut_nonlinear_sim/h2lib/bem3d_cy.pyx'],
        **ext_opts
    ),
    Extension(
        name='cmut_nonlinear_sim.h2lib.helmholtzbem3d_cy',
        sources=['cmut_nonlinear_sim/h2lib/helmholtzbem3d_cy.pyx'],
        **ext_opts
    ),
]


setup(
    name='cmut-nonlinear-sim',
    version='0.1',
    ext_modules=cythonize(ext_modules),
    packages=find_packages(),
    include_dirs=['include'],
    install_requires=[
    ],
    setup_requires=[
        'setuptools>=18.0', 
        'cython>=0.25']
)

