''''''
import numpy as np
import numpy.linalg
from cnld import abstract, mesh, util
from scipy.constants import epsilon_0 as e_0
import numba

eps = np.finfo(np.float64).eps


def bm_ndarray(geom, M, K):
    '''
    Damping matrix based on Rayleigh damping for damping ratios at two
    frequencies.
    '''
    fa = geom.damping_freq_a
    fb = geom.damping_freq_b
    za = geom.damping_ratio_a
    zb = geom.damping_ratio_b

    omga = 2 * np.pi * fa
    omgb = 2 * np.pi * fb

    # solve for alpha and beta
    A = 1 / 2 * np.array([[1 / omga, omga], [1 / omgb, omgb]])
    alpha, beta = linalg.inv(A).dot([za, zb])

    return alpha * M + beta * K


def beigm_ndarray(grid, geom, M, K):
    '''
    Damping matrix based on Rayleigh damping for damping ratios at two modal
    frequencies.
    '''
    ma = geom.damping_mode_a
    mb = geom.damping_mode_b
    za = geom.damping_ratio_a
    zb = geom.damping_ratio_b

    # determine eigenfrequencies of membrane
    eigf, _ = mem_eig(grid, geom)
    omga = eigf[ma] * 2 * np.pi
    omgb = eigf[mb] * 2 * np.pi

    # solve for alpha and beta
    A = 1 / 2 * np.array([[1 / omga, omga], [1 / omgb, omgb]])
    alpha, beta = linalg.inv(A).dot([za, zb])

    return alpha * M + beta * K


def mem_eig(grid, geom):
    '''
    Returns the eigenfrequency (in Hz) and eigenmodes of a membrane.
    '''
    ob = grid.on_boundary

    M = mm_ndarray(grid, geom, mu=0.5)
    K = km_ndarray(grid, geom)
    w, v = linalg.eig(linalg.inv(M).dot(K)[np.ix_(~ob, ~ob)])

    idx = np.argsort(np.sqrt(np.abs(w)))
    eigf = np.sqrt(np.abs(w))[idx] / (2 * np.pi)
    eigv = v[:, idx]

    return eigf, eigv