'''
'''
__all__ = ['x_stat_vec_np']

import numpy as np
import numpy.linalg
from scipy.constants import epsilon_0 as e_0
from . import _stiffness as stiffness, _rhs as rhs


def x_stat_vec_np(grid, geom, vdc, type='bpt', atol=1e-10, maxiter=100):
    '''
    Static deflection of membrane calculated via fixed-point iteration.
    '''

    def pes(v, x, g_eff):
        return -e_0 / 2 * v**2 / (g_eff + x)**2

    K = stiffness.k_mat_np(grid, geom, type=type)
    g_eff = geom.gap + geom.isolation / geom.permittivity
    F = rhs.p_vec_np(grid, geom, 1)
    Kinv = np.linalg.inv(K)
    nnodes = K.shape[0]
    x0 = np.zeros(nnodes)

    for i in range(maxiter):
        x0_new = Kinv.dot(F * pes(vdc, x0, g_eff))

        if np.max(np.abs(x0_new - x0)) < atol:
            is_collapsed = False
            return x0_new, is_collapsed

        x0 = x0_new

    is_collapsed = True
    return x0, is_collapsed
