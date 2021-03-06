'''
Cnld setup script.
'''
from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
import os

import numpy as np

# yapf: disable
ext_opts = {
    'include_dirs': ['include', np.get_include()],
    # 'libraries':['h2', 'blas', 'lapack', 'gfortran'],
    'libraries': ['h2', 'gfortran', 'openblas', 'omp'],
    'library_dirs': ['lib'],
    'language': 'c',
    'extra_compile_args': ['-fPIC', '-DUSE_BLAS', '-DUSE_COMPLEX', '-DUSE_OPENMP', '-Wno-strict-prototypes'],
    # 'extra_objects':['./lib/libopenblas.a']
}

ext_modules = [
    Extension(
        name='cnld.h2lib.basic',
        sources=['cnld/h2lib/basic.pyx'],
        **ext_opts
    ),
    Extension(
        name='cnld.h2lib.avector',
        sources=['cnld/h2lib/avector.pyx'],
        **ext_opts
    ),
    Extension(
        name='cnld.h2lib.amatrix',
        sources=['cnld/h2lib/amatrix.pyx'],
        **ext_opts
    ),
    Extension(
        name='cnld.h2lib.cluster',
        sources=['cnld/h2lib/cluster.pyx'],
        **ext_opts
    ),
    Extension(
        name='cnld.h2lib.block',
        sources=['cnld/h2lib/block.pyx'],
        **ext_opts
    ),
    Extension(
        name='cnld.h2lib.surface3d',
        sources=['cnld/h2lib/surface3d.pyx'],
        **ext_opts
    ),
    Extension(
        name='cnld.h2lib.macrosurface3d',
        sources=['cnld/h2lib/macrosurface3d.pyx'],
        **ext_opts
    ),
    Extension(
        name='cnld.h2lib.rkmatrix',
        sources=['cnld/h2lib/rkmatrix.pyx'],
        **ext_opts
    ),
    Extension(
        name='cnld.h2lib.hmatrix',
        sources=['cnld/h2lib/hmatrix.pyx'],
        **ext_opts
    ),
    Extension(
        name='cnld.h2lib.bem3d',
        sources=['cnld/h2lib/bem3d.pyx'],
        **ext_opts
    ),
    Extension(
        name='cnld.h2lib.helmholtzbem3d',
        sources=['cnld/h2lib/helmholtzbem3d.pyx'],
        **ext_opts
    ),
    Extension(
        name='cnld.h2lib.truncation',
        sources=['cnld/h2lib/truncation.pyx'],
        **ext_opts
    ),
    Extension(
        name='cnld.h2lib.krylovsolvers',
        sources=['cnld/h2lib/krylovsolvers.pyx'],
        **ext_opts
    ),
    Extension(
        name='cnld.h2lib.harith',
        sources=['cnld/h2lib/harith.pyx'],
        **ext_opts
    ),
    Extension(
        name='cnld.h2lib.factorizations',
        sources=['cnld/h2lib/factorizations.pyx'],
        **ext_opts
    ),
    Extension(
        name='cnld.h2lib.sparsematrix',
        sources=['cnld/h2lib/sparsematrix.pyx'],
        **ext_opts
    ),
    Extension(
        name='cnld.h2lib.matrixnorms',
        sources=['cnld/h2lib/matrixnorms.pyx'],
        **ext_opts
    ),
    Extension(
        name='cnld.simulation',
        sources=['cnld/simulation.pyx'],
        **ext_opts
    ),
    Extension(
        name='cnld.bem_pressure',
        sources=['cnld/bem_pressure.pyx'],
        **ext_opts
    ),
    Extension(
        name='cnld.simulation',
        sources=['cnld/simulation.pyx'],
        **ext_opts
    ),
]

setup(
    name='cnl-dyna',
    version='0.9',
    ext_modules=cythonize(ext_modules),
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'cnld = cnld.cli2:main'
        ]
    },
    setup_requires=[
        'setuptools>=18.0',
        'cython>=0.25',
    ],
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'tqdm',
        'pandas',
        'namedlist',
        'jupyterlab',
    ]
)
