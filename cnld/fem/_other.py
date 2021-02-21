'''
'''
import numpy as np
import numpy.linalg


def mem_static_x_vector(mem, refn, vdc, type='bpt', atol=1e-10, maxiter=100):
    '''
    Static deflection of membrane calculated via fixed-point iteration.
    '''
    def pes(v, x, g_eff):
        return -e_0 / 2 * v**2 / (g_eff + x)**2

    if isinstance(mem, abstract.SquareCmutMembrane):
        amesh = mesh.square(mem.length_x, mem.length_y, refn)
    else:
        amesh = mesh.circle(mem.radius, refn)

    K = mem_k_matrix(mem, refn, type=type)
    g_eff = mem.gap + mem.isolation / mem.permittivity
    F = mem_f_vector(mem, refn, 1)
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
