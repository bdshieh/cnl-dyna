## bem.py ##

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from scipy import sparse as sps, linalg
from scipy.io import loadmat

from . compressed_formats import ZHMatrix, ZFullMatrix, MbkSparseMatrix, MbkFullMatrix
from . mesh import Mesh, square, circle
from . import util


@util.memoize
def mem_z_matrix(mesh, k, *args, **kwargs):
    return ZFullMatrix(mesh, k, *args, **kwargs).data


def z_from_abstract(array, k, refn, format='HFormat', *args, **kwargs):
    mesh = Mesh.from_abstract(array, refn)
    return z_from_mesh(mesh, k, format, *args, **kwargs)


def z_from_mesh(mesh, k, format='HFormat', *args, **kwargs):
    if format.lower() in ['hformat', 'h']:
        return ZHMatrix(mesh, k, *args, **kwargs)
    else:
        return ZFullMatrix(mesh, k, *args, **kwargs)


def z_linear_operators(array, f, c, refn, rho=1000., *args, **kwargs):

    k = 2 * np.pi * f / c
    omg = 2 * np.pi * f

    Z = z_from_abstract(array, k, refn, *args, **kwargs)
    Z_LU = Z.lu()
    mesh = Mesh.from_abstract(array, refn=refn)
    ob = mesh.on_boundary
    nnodes = len(mesh.vertices)

    def mv(x):
        x[ob] = 0
        p = Z * x
        p[ob] = 0
        return -omg**2 * rho * 2 * p
    linop = sps.linalg.LinearOperator((nnodes, nnodes), dtype=np.complex128, matvec=mv)

    def inv_mv(x):
        x[ob] = 0
        p = Z_LU._triangularsolve(x)
        p[ob] = 0
        return -omg**2 * rho * 2 * p
    linop_inv = sps.linalg.LinearOperator((nnodes, nnodes), dtype=np.complex128, matvec=inv_mv)
    
    return linop, linop_inv


    
