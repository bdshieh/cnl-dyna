'''Routines for the boundary element method.'''

import cmath

import numpy as np
import scipy as sp
from cnld import abstract, mesh, util
from cnld.compressed_formats import ZFullMatrix, ZHMatrix
from scipy import linalg
from scipy import sparse as sps
from scipy.io import loadmat


@util.memoize
def mem_z_matrix(mem, refn, k, *args, **kwargs):
    '''
    Impedance matrix in FullFormat for a membrane.

    Parameters
    ----------
    mem : [type]
        [description]
    refn : [type]
        [description]
    k : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    '''
    if isinstance(mem, abstract.SquareCmutMembrane):
        amesh = mesh.square(mem.length_x, mem.length_y, refn)
    else:
        amesh = mesh.circle(mem.radius, refn)

    return np.array(ZFullMatrix(amesh, k, *args, **kwargs).data)


def array_z_matrix(array, refn, k, format='HFormat', *args, **kwargs):
    '''
    Impedance matrix in either FullFormat or HFormat for an array.

    Parameters
    ----------
    array : [type]
        [description]
    refn : [type]
        [description]
    k : [type]
        [description]
    format : str, optional
        [description], by default 'HFormat'

    Returns
    -------
    [type]
        [description]
    '''
    amesh = mesh.Mesh.from_abstract(array, refn)

    if format.lower() in ['hformat', 'h']:
        return ZHMatrix(amesh, k, *args, **kwargs)
    else:
        return ZFullMatrix(amesh, k, *args, **kwargs)


def array_z_linop(array, refn, f, c, rho, *args, **kwargs):
    '''
    Impedance matrix linear operators for an array.

    Parameters
    ----------
    array : [type]
        [description]
    refn : [type]
        [description]
    f : [type]
        [description]
    c : [type]
        [description]
    rho : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    '''
    k = 2 * np.pi * f / c
    omg = 2 * np.pi * f

    Z = array_z_matrix(array, refn, k, *args, **kwargs)
    Z_LU = Z.lu()

    amesh = mesh.Mesh.from_abstract(array, refn)
    ob = amesh.on_boundary
    nnodes = len(amesh.vertices)

    def mv(x):
        x[ob] = 0
        p = Z * x
        p[ob] = 0
        return -omg**2 * rho * 2 * p

    linop = sps.linalg.LinearOperator((nnodes, nnodes),
                                      dtype=np.complex128,
                                      matvec=mv)

    def inv_mv(x):
        x[ob] = 0
        p = Z_LU._triangularsolve(x)
        p[ob] = 0
        return -omg**2 * rho * 2 * p

    linop_inv = sps.linalg.LinearOperator((nnodes, nnodes),
                                          dtype=np.complex128,
                                          matvec=inv_mv)

    return linop, linop_inv
