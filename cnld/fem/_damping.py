''''''
__all__ = ['b_mat_np', 'b_eig_mat_np', 'geom_eig']

import numpy as np
import numpy.linalg
# import numba
from . import _mass as mass, _stiffness as stiffness


def b_mat_np(geom, M, K):
    '''
    Damping matrix based on Rayleigh damping for damping ratios at two
    frequencies.
    '''
    fa = geom.damping_freq1
    fb = geom.damping_freq2
    za = geom.damping_ratio1
    zb = geom.damping_ratio2

    omga = 2 * np.pi * fa
    omgb = 2 * np.pi * fb

    # solve for alpha and beta
    A = 1 / 2 * np.array([[1 / omga, omga], [1 / omgb, omgb]])
    alpha, beta = np.linalg.inv(A).dot([za, zb])

    return alpha * M + beta * K


def b_eig_mat_np(grid, geom, M, K):
    '''
    Damping matrix based on Rayleigh damping for damping ratios at two modal
    frequencies.
    '''
    ma = geom.damping_mode1
    mb = geom.damping_mode2
    za = geom.damping_ratio1
    zb = geom.damping_ratio2

    # determine eigenfrequencies of membrane
    eigf, _ = geom_eig(grid, geom)
    omga = eigf[ma] * 2 * np.pi
    omgb = eigf[mb] * 2 * np.pi

    # solve for alpha and beta
    A = 1 / 2 * np.array([[1 / omga, omga], [1 / omgb, omgb]])
    alpha, beta = np.linalg.inv(A).dot([za, zb])

    return alpha * M + beta * K


def geom_eig(grid, geom):
    '''
    Returns the eigenfrequency (in Hz) and eigenmodes of a membrane.
    '''
    ob = grid.on_boundary

    M = mass.m_mat_np(grid, geom)
    K = stiffness.k_mat_np(grid, geom)
    w, v = np.linalg.eig(np.linalg.inv(M).dot(K)[np.ix_(~ob, ~ob)])

    idx = np.argsort(np.sqrt(np.abs(w)))
    eigf = np.sqrt(np.abs(w))[idx] / (2 * np.pi)
    eigv = v[:, idx]

    eigv_full = np.zeros((grid.nvertices, len(eigf)))
    eigv_full[~ob, :] = eigv

    return eigf, eigv_full
