## bem.py ##

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from scipy import sparse as sps, linalg
from scipy.io import loadmat

from . compressed_formats import ZMatrix, MBKMatrix
# from . import h2lib
from . mesh import from_spec



def isiterable(obj):
    try:
        iter(obj)
        return True
    except TypeError:
        return False


def M_matrix(n, rho, h):

    if not isiterable(rho):
        rho = [rho,]
    if not isiterable(h):
        h = [h,]
    mass = sum([x * y for x, y in zip(rho, h)])

    return sps.csr_matrix(sps.eye(n) * mass)


def B_matrix(n, att):
    return sps.csr_matrix(sps.eye(n) * att)


def K_matrix_comsol(nmem, file, refn):
    ''''''
    # K = linalg.inv(loadmat(file)['x'].T)
    with np.load(file) as root:
        Ks = root['K']
        refns = root['refn']
    
    idx = np.argmin(np.abs(refns - refn))
    K = Ks[idx]

    return sps.csr_matrix(sps.block_diag([K for i in range(nmem)]))


def K_matrix_pzflex():
    pass

    
def MBK_matrix(f, n, nmem, rho, h, att, kfile, compress=True):
    
    M = M_matrix(n, rho, h)
    B = B_matrix(n, att)
    K = K_matrix_comsol(nmem, kfile)

    omg = 2 * np.pi * f 
    MBK = -(omg ** 2) * M + 1j * omg * B + K

    if compress:
        return MBKMatrix(MBK)
    return MBK


def Z_matrix(format, mesh, k, *args, **kwargs):
    return ZMatrix(format, mesh, k, *args, **kwargs)



# for frequency each f, create mesh from spec

# from mesh, generate M, B, K as Scipy sparse matrices
# convert M, B, K to MBK in compressed sparseformat

# generate Z in compressed hformat
# perform G = MBK + Z by converting MBK to hformat and adding

# decompose LU = G
# solve x(f) using LU for lumped forcings
# calculate time impulse response using ifft

