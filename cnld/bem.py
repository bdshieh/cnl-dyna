## bem.py ##

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from scipy import sparse as sps, linalg
from scipy.io import loadmat

from . compressed_formats2 import ZHMatrix, ZFullMatrix, MbkSparseMatrix, MbkFullMatrix
from . mesh import Mesh, square, circle
from . import util


# def isiterable(obj):
#     try:
#         iter(obj)
#         return True
#     except TypeError:
#         return False


# @util.memoize
# def mem_m_matrix(n, rho, h):

#     if not isiterable(rho):
#         rho = [rho,]
#     if not isiterable(h):
#         h = [h,]
#     mass = sum([x * y for x, y in zip(rho, h)])

#     return sps.csr_matrix(sps.eye(n) * mass)


# @util.memoize
# def mem_b_matrix(n, att):
#     return sps.csr_matrix(sps.eye(n) * att)


# @util.memoize
# def mem_k_matrix(file, refn):
#     with np.load(file) as npf:
#         idx = np.argmax(npf['refns'] == refn)
#         K = npf['Ks'][idx]
#     return K


# @util.memoize
# def get_nvert_square_membrane(xl, yl, refn):
#     mesh = square(xl, yl, refn=refn)
#     return len(mesh.vertices)


# @util.memoize
# def get_nvert_circle_membrane(xl, yl, refn):
#     raise NotImplementedError


# def mbk_from_abstract(array, f, refn, format='SparseFormat'):

#     omg = 2 * np.pi * f
#     blocks = []
#     for elem in array.elements:
#         for mem in elem.membranes:
#             n = get_nvert_square_membrane(mem.length_x, mem.length_y, refn)
#             M = mem_m_matrix(n, mem.density, mem.thickness)
#             B = mem_b_matrix(n, mem.att_mech)
#             K = mem_k_matrix(mem.kmat_file, refn) # / 1e9  ###### TEMPORARY

#             block = -(omg ** 2) * M + 1j * omg * B + K
#             blocks.append(block)
    
#     if format.lower() in ['sparse', 'sparseformat']:
#         return MbkSparseMatrix(sps.csr_matrix(sps.block_diag(blocks)))
#     else:
#         return MbkFullMatrix(sps.block_diag(blocks).todense())


# def mbk_from_mesh():
#     raise NotImplementedError


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
    mesh = Mesh.from_abstract(array)
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

