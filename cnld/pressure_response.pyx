'''
'''
from __future__ import division

import numpy as np
cimport numpy as np
import cython
cimport cython
cimport libc.math

from cnld import mesh, impulse_response, database, abstract


cdef extern from "complex.h":
    double complex cexp(double complex)


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


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double complex kernel_helmholtz(double k, double x1, double y1, double z1, double x2, double y2, double z2):
    '''
    Helmholtz kernel for acoustic waves.
    '''
    cdef double r

    r = libc.math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
    return cexp(-1j * k * r) / (4 * np.pi * r)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray mesh_pres_vector(double[:,:] vertices, unsigned int[:,:] triangles, double[:] triangle_areas, 
    list r, double k, double c, double rho, int gn=2):
    '''
    Frequency-domain pressure calculation vector from surface mesh.
    '''
    cdef int nnodes = vertices.shape[0]
    cdef double complex[:] p = np.zeros(nnodes, dtype=np.complex128)
    cdef double x, y, z
    cdef int tt

    x, y, z = r

    gr, gw = gauss_quadrature(gn)

    for tt in range(len(triangles)):
        tri = triangles[tt,:]
        x1, y1 = vertices[tri[0],:2]
        x2, y2 = vertices[tri[1],:2]
        x3, y3 = vertices[tri[2],:2]

        da = triangle_areas[tt]

        for (xi, eta), w in zip(gr, gw):
            
            xs = x1 * (1 - xi - eta) + x2 * xi + x3 * eta
            ys = y1 * (1 - xi - eta) + y2 * xi + y3 * eta
            zs = 0

            cfac = w * kernel_helmholtz(k, xs, ys, zs, x, y, z) * da
            p[tri[0]] += (1 - xi - eta) * cfac
            p[tri[1]] += xi * cfac
            p[tri[2]] += eta * cfac

    return -(k * c)**2 * rho * 2 * np.array(p)


def array_patch_pres_imp_resp(array, refn, db_file, r, c, rho, use_kkr=False, interp=2):
    '''
    '''
    # read database
    freqs, pnfr, nodes = database.read_patch_to_node_freq_resp(db_file)

    patches = abstract.get_patches_from_array(array)
    amesh = mesh.Mesh.from_abstract(array, refn)

    sfr = np.zeros((len(patches), len(freqs)), dtype=np.complex128)

    for i, f in enumerate(freqs):
        omg = 2 *np.pi * f
        k = omg / c

        p_vector = mesh_pres_vector(amesh.vertices, amesh.triangles, amesh.triangle_areas, r, k, c, rho)

        for j in range(len(patches)):
            disp = pnfr[j,:,i]
            sfr[j,i] = p_vector.dot(disp)

    sir_t, sir = impulse_response.fft_to_sir(freqs, sfr, interp=interp, axis=1, use_kkr=use_kkr)

    return sir_t, sir


def press_resp(array, refn, db_file, pes, r, c, rho, use_kkr=False, interp=2):

    sir_t, sir = array_patch_pres_imp_resp(array, refn, db_file, r, c, rho, interp=interp, use_kkr=use_kkr)
    sir = sir.T

    dt = sir_t[1] - sir_t[0]

    ppatch = []
    for i in range(pes.shape[1]):
        ppatch.append(np.convolve(pes[:,i], sir[:,i]) * dt)

    return np.sum(ppatch, axis=0)