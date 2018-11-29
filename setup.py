## interaction / setup.py
'''
'''
from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize

# import numpy as np
# import os


ext_opts = {
    'include_dirs':['include'],
    # 'libraries':['h2', 'blas', 'lapack', 'gfortran'],
    'libraries':['h2', 'gfortran', 'openblas', 'omp'],
    'library_dirs':['lib'],
    'language':'c',
    'extra_compile_args':['-fPIC', '-DUSE_BLAS', '-DUSE_COMPLEX', '-DUSE_OPENMP', '-Wno-strict-prototypes'],
    # 'extra_objects':['./lib/libopenblas.a']
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
    Extension(
        name='cmut_nonlinear_sim.h2lib.truncation_cy',
        sources=['cmut_nonlinear_sim/h2lib/truncation_cy.pyx'],
        **ext_opts
    ),
    Extension(
        name='cmut_nonlinear_sim.h2lib.krylovsolvers_cy',
        sources=['cmut_nonlinear_sim/h2lib/krylovsolvers_cy.pyx'],
        **ext_opts
    ),
    Extension(
        name='cmut_nonlinear_sim.h2lib.harith_cy',
        sources=['cmut_nonlinear_sim/h2lib/harith_cy.pyx'],
        **ext_opts
    ),
    Extension(
        name='cmut_nonlinear_sim.h2lib.factorizations_cy',
        sources=['cmut_nonlinear_sim/h2lib/factorizations_cy.pyx'],
        **ext_opts
    ),
    Extension(
        name='cmut_nonlinear_sim.h2lib.sparsematrix_cy',
        sources=['cmut_nonlinear_sim/h2lib/sparsematrix_cy.pyx'],
        **ext_opts
    ),
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
        'cython>=0.25',
        'numpy',
        'scipy',
        'matplotlib',
        # 'openmp',
        # 'libgfortran',
        'tqdm']
)

