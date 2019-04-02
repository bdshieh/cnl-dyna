## bem.py ##

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from scipy import sparse as sps, linalg
from scipy.io import loadmat

from . compressed_formats import ZHMatrix, ZFullMatrix, MbkSparseMatrix, MbkFullMatrix
from . mesh import Mesh, square, circle
from . import util
from cnld import impulse_response, database, abstract


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


def pressure_from_abstract_and_db(array, refn, db_file, r, c, rho, use_kkr=True, mult=5):
    '''
    '''
    # read database
    freqs, pnfr, nodes = database.read_patch_to_node_freq_resp(db_file)

    patches = abstract.get_patches_from_array(array)
    amesh = Mesh.from_abstract(array, refn)

    sfr = np.zeros((len(patches), len(freqs)), dtype=np.complex128)

    for i, f in enumerate(freqs):
        omg = 2 *np.pi * f
        k = omg / c

        for j in range(len(patches)):
            disp = pnfr[j,:,i]
            sfr[j,i] = pressurefd(amesh, disp, r, k, c, rho)

    sir_t, sir = impulse_response.fft_to_fir(freqs, sfr, mult=mult, axis=1, use_kkr=use_kkr)

    return sir_t, sir


def gauss_quadrature(n, type=1):
    '''
    Gaussian quadrature rules for triangular element surface integrals.
    '''
    if n == 1:
        return [[1/3, 1/3]], [1,]
    elif n == 2:
        if type == 1:
            return [[1/6, 1/6], [2/3, 1/6], [1/6, 2/3]], [1/3, 1/3, 1/3]
        elif type == 2:
            return [[0, 1/2], [1/2, 0], [1/2, 1/2]], [1/3, 1/3, 1/3]
    elif n == 3:
        if type == 1:
            return [[1/3, 1/3], [1/5, 3/5], [1/5, 1/5], [3/5, 1/5]] ,[-27/48, 25/48, 25/48, 25/48]
        elif type == 2:
            return [[1/3, 1/3], [2/15, 11/15], [2/15, 2/15], [11/15, 2/15]] ,[-27/48, 25/48, 25/48, 25/48]


def pressurefd(amesh, disp, r, k, c, rho, gn=2):
    '''
    Frequency-domain pressure calculation from surface mesh.
    '''
    kernel = helmholtz_kernel
    nodes = amesh.vertices
    triangles = amesh.triangles
    triangle_areas = amesh.triangle_areas

    x, y, z = r

    gr, gw = gauss_quadrature(gn)

    p = 0

    for tt in range(len(triangles)):
        tri = triangles[tt,:]
        x1, y1 = nodes[tri[0],:2]
        x2, y2 = nodes[tri[1],:2]
        x3, y3 = nodes[tri[2],:2]

        u1 = disp[tri[0]]
        u2 = disp[tri[1]]
        u3 = disp[tri[2]]

        da = triangle_areas[tt]

        for (xi, eta), w in zip(gr, gw):
            
            xs = x1 * (1 - xi - eta) + x2 * xi + x3 * eta
            ys = y1 * (1 - xi - eta) + y2 * xi + y3 * eta
            zs = 0

            u = u1 * (1 - xi - eta) + u2 * xi + u3 * eta

            p += w * u * kernel(k, xs, ys, zs, x, y, z)
            
        p *= da

    return -(k * c)**2 * rho * 2 * p


def helmholtz_kernel(k, x1, y1, z1, x2, y2, z2):
    '''
    Helmholtz kernel for acoustic waves.
    '''
    r = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
    return np.exp(-1j * k * r) / r

