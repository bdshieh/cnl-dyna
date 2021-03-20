'''
'''
__all__ = ['x_stat_vec_np']

import numpy as np
import numpy.linalg
from scipy.constants import epsilon_0 as e_0
from . import _stiffness as stiffness, _rhs as rhs
# import numba


def x_stat_vec_np(grid,
                  geom,
                  vdc,
                  type='bpt',
                  use_contact=False,
                  atol=1e-10,
                  maxiter=100):
    '''
    Static deflection of membrane calculated via fixed-point iteration.
    '''

    def pes(v, x, g_eff):
        return -e_0 / 2 * v**2 / (g_eff + x)**2

    K = stiffness.k_mat_np(grid, geom, type=type)
    g_eff = geom.gap + geom.isol_thickness / geom.eps_r
    P = rhs.p_vec_np(grid, 1)
    Kinv = np.linalg.inv(K)
    nnodes = K.shape[0]
    x0 = np.zeros(nnodes)

    for i in range(maxiter):
        x0_new = Kinv.dot(P * pes(vdc, x0, g_eff))

        if np.max(np.abs(x0_new - x0)) < atol:
            is_collapsed = False
            return x0_new, is_collapsed

        x0 = x0_new

    if use_contact:
        x0[x0 < -geom.gap] = -geom.gap
    is_collapsed = True
    return x0, is_collapsed


# def x_stat_cont_vec_np(grid, geom, vdc, type='bpt', atol=1e-10, maxiter=100):
#     '''
#     Static deflection of membrane calculated via fixed-point iteration.
#     '''

#     def pes(v, x, g_eff):
#         return -e_0 / 2 * v**2 / (g_eff + x)**2

#     # @numba.vectorize
#     # def p_cont_spr(x, z0, k, n):
#     #     if x >= z0:
#     #         return 0
#     #     return k * (z0 - x)**n

#     K = stiffness.k_mat_np(grid, geom, type=type)
#     g_eff = geom.gap + geom.isol_thickness / geom.eps_r
#     P = rhs.p_vec_np(grid, 1)

#     Kinv = np.linalg.inv(K)
#     nnodes = K.shape[0]
#     x0 = np.zeros(nnodes)

#     for i in range(maxiter):
#         # x0_new = Kinv.dot(
#         #     P * pes(vdc, x0, g_eff) +
#         #     p_cont_spr(x0, geom.contact_z0, geom.contact_k, geom.contact_n))
#         x0_new = Kinv.dot(P * pes(vdc, x0, g_eff))
#         x0_new[x0_new < -geom.gap] = -geom.gap

#         if np.max(np.abs(x0_new - x0)) < atol:
#             is_collapsed = False
#             return x0_new, is_collapsed

#         x0 = x0_new

#     is_collapsed = True
#     return x0, is_collapsed
