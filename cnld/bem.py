'''
'''
import numpy as np
import scipy as sp
from scipy import sparse as sps, linalg
from scipy.io import loadmat
import cmath

from cnld.compressed_formats import ZHMatrix, ZFullMatrix
from cnld import util, mesh, impulse_response, database, abstract


@util.memoize
def mem_z_matrix(mem, refn, k, *args, **kwargs):
    '''
    '''
    if isinstance(mem, abstract.SquareCmutMembrane):
        amesh = mesh.square(mem.length_x, mem.length_y, refn)
    else:
        amesh = mesh.circle(mem.radius, refn)

    return np.array(ZFullMatrix(amesh, k, *args, **kwargs).data)


def array_z_matrix(array, refn, k, format='HFormat', *args, **kwargs):
    '''
    '''
    amesh = mesh.Mesh.from_abstract(array, refn)

    if format.lower() in ['hformat', 'h']:
        return ZHMatrix(amesh, k, *args, **kwargs)
    else:
        return ZFullMatrix(amesh, k, *args, **kwargs)


def z_linear_operators(array, refn, f, c, rho, *args, **kwargs):
    '''
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


# def gauss_quadrature(n, type=1):
#     '''
#     Gaussian quadrature rules for triangular element surface integrals.
#     '''
#     if n == 1:
#         return [[1/3, 1/3]], [1,]
#     elif n == 2:
#         if type == 1:
#             return [[1/6, 1/6], [2/3, 1/6], [1/6, 2/3]], [1/3, 1/3, 1/3]
#         elif type == 2:
#             return [[0, 1/2], [1/2, 0], [1/2, 1/2]], [1/3, 1/3, 1/3]
#     elif n == 3:
#         if type == 1:
#             return [[1/3, 1/3], [1/5, 3/5], [1/5, 1/5], [3/5, 1/5]] ,[-27/48, 25/48, 25/48, 25/48]
#         elif type == 2:
#             return [[1/3, 1/3], [2/15, 11/15], [2/15, 2/15], [11/15, 2/15]] ,[-27/48, 25/48, 25/48, 25/48]

# # def kernel_helmholtz(k, x1, y1, z1, x2, y2, z2):
# #     '''
# #     Helmholtz kernel for acoustic waves.
# #     '''
# #     r = cmath.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
# #     return cmath.exp(-1j * k * r) / (4 * cmath.pi * r)

# cdef extern from "complex.h":
#     double complex cexp(double complex)

# @cython.boundscheck(False)
# @cython.wraparound(False)
# cpdef kernel_helmholtz(double k, double x1, double y1, double z1, double x2, double y2, double z2):
#     '''
#     Helmholtz kernel for acoustic waves.
#     '''
#     cdef double r

#     r = libc.math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
#     return cexp(-1j * k * r) / (4 * np.pi * r)

# def mesh_pres_vector(amesh, r, k, c, rho, gn=2):
#     '''
#     Frequency-domain pressure calculation vector from surface mesh.
#     '''
#     kernel = kernel_helmholtz
#     nodes = amesh.vertices
#     nnodes = len(nodes)
#     triangles = amesh.triangles
#     triangle_areas = amesh.triangle_areas

#     x, y, z = r

#     gr, gw = gauss_quadrature(gn)

#     p = np.zeros(nnodes, dtype=np.complex128)

#     for tt in range(len(triangles)):
#         tri = triangles[tt,:]
#         x1, y1 = nodes[tri[0],:2]
#         x2, y2 = nodes[tri[1],:2]
#         x3, y3 = nodes[tri[2],:2]

#         da = triangle_areas[tt]

#         for (xi, eta), w in zip(gr, gw):

#             xs = x1 * (1 - xi - eta) + x2 * xi + x3 * eta
#             ys = y1 * (1 - xi - eta) + y2 * xi + y3 * eta
#             zs = 0

#             cfac = w * kernel(k, xs, ys, zs, x, y, z) * da
#             p[tri[0]] += (1 - xi - eta) * cfac
#             p[tri[1]] += xi * cfac
#             p[tri[2]] += eta * cfac

#     return -(k * c)**2 * rho * 2 * p

# def array_patch_pres_imp_resp(array, refn, db_file, r, c, rho, use_kkr=False, mult=2):
#     '''
#     '''
#     # read database
#     freqs, pnfr, nodes = database.read_patch_to_node_freq_resp(db_file)

#     patches = abstract.get_patches_from_array(array)
#     amesh = mesh.Mesh.from_abstract(array, refn)

#     sfr = np.zeros((len(patches), len(freqs)), dtype=np.complex128)

#     for i, f in enumerate(freqs):
#         omg = 2 *np.pi * f
#         k = omg / c

#         p_vector = mesh_pres_vector(amesh, r, k, c, rho)

#         for j in range(len(patches)):
#             disp = pnfr[j,:,i]
#             sfr[j,i] = p_vector.dot(disp)

#     sir_t, sir = impulse_response.fft_to_sir(freqs, sfr, mult=mult, axis=1, use_kkr=use_kkr)

#     return sir_t, sir