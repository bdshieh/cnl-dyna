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
        name='cnls.h2lib.basic_cy',
        sources=['cnls/h2lib/basic_cy.pyx'],
        **ext_opts
    ),
    Extension(
        name='cnls.h2lib.avector_cy',
        sources=['cnls/h2lib/avector_cy.pyx'],
        **ext_opts
    ),
    Extension(
        name='cnls.h2lib.amatrix_cy',
        sources=['cnls/h2lib/amatrix_cy.pyx'],
        **ext_opts
    ),
    Extension(
        name='cnls.h2lib.cluster_cy',
        sources=['cnls/h2lib/cluster_cy.pyx'],
        **ext_opts
    ),
    Extension(
        name='cnls.h2lib.block_cy',
        sources=['cnls/h2lib/block_cy.pyx'],
        **ext_opts
    ),
    Extension(
        name='cnls.h2lib.surface3d_cy',
        sources=['cnls/h2lib/surface3d_cy.pyx'],
        **ext_opts
    ),
    Extension(
        name='cnls.h2lib.macrosurface3d_cy',
        sources=['cnls/h2lib/macrosurface3d_cy.pyx'],
        **ext_opts
    ),
    Extension(
        name='cnls.h2lib.rkmatrix_cy',
        sources=['cnls/h2lib/rkmatrix_cy.pyx'],
        **ext_opts
    ),
    Extension(
        name='cnls.h2lib.hmatrix_cy',
        sources=['cnls/h2lib/hmatrix_cy.pyx'],
        **ext_opts
    ),
    Extension(
        name='cnls.h2lib.bem3d_cy',
        sources=['cnls/h2lib/bem3d_cy.pyx'],
        **ext_opts
    ),
    Extension(
        name='cnls.h2lib.helmholtzbem3d_cy',
        sources=['cnls/h2lib/helmholtzbem3d_cy.pyx'],
        **ext_opts
    ),
    Extension(
        name='cnls.h2lib.truncation_cy',
        sources=['cnls/h2lib/truncation_cy.pyx'],
        **ext_opts
    ),
    Extension(
        name='cnls.h2lib.krylovsolvers_cy',
        sources=['cnls/h2lib/krylovsolvers_cy.pyx'],
        **ext_opts
    ),
    Extension(
        name='cnls.h2lib.harith_cy',
        sources=['cnls/h2lib/harith_cy.pyx'],
        **ext_opts
    ),
    Extension(
        name='cnls.h2lib.factorizations_cy',
        sources=['cnls/h2lib/factorizations_cy.pyx'],
        **ext_opts
    ),
    Extension(
        name='cnls.h2lib.sparsematrix_cy',
        sources=['cnls/h2lib/sparsematrix_cy.pyx'],
        **ext_opts
    ),
]


setup(
    name='cnls',
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

