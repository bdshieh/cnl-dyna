'''Routines for the boundary element method.'''
__all__ = [
    'z_mat_np', 'z_mat_fm', 'z_mat_hm', 'z_mat_hm_from_grid',
    'z_mat_fm_from_grid', 'z_mat_np_from_grid'
]
import numpy as np
import numba
import math
from timeit import default_timer as timer
from cnld.matrix import H2FullMatrix, H2HMatrix
from cnld import database, impulse_response
from .h2lib import *


def z_mat_np_from_grid(grid, k, **kwargs):
    '''
    Impedance matrix in FullFormat for a membrane.
    '''
    return np.array(z_mat_fm_from_grid(grid, k, **kwargs).data)


def z_mat_fm_from_grid(grid, k, basis='linear', q_reg=2, q_sing=4):
    '''
    Impedance matrix in full format.
    '''

    if basis.lower() in ['constant']:
        _basis = basisfunctionbem3d.CONSTANT
    elif basis.lower() in ['linear']:
        _basis = basisfunctionbem3d.LINEAR
    else:
        raise ValueError

    bem = new_slp_helmholtz_bem3d(k, grid._mesh.surface3d, q_reg, q_sing,
                                  _basis, _basis)

    Z = H2FullMatrix.zeros((grid.nvertices, grid.nvertices))

    start = timer()
    assemble_bem3d_amatrix(bem, Z._mat)
    time_assemble = timer() - start

    Z._time_assemble = time_assemble

    return Z


def z_mat_hm_from_grid(grid,
                       k,
                       basis='linear',
                       m=4,
                       q_reg=2,
                       q_sing=4,
                       aprx='paca',
                       admis='2',
                       eta=1.0,
                       eps_aca=1e-2,
                       strict=False,
                       clf=16,
                       rk=0):
    '''
    Impedance matrix in hierarchical format.
    '''

    if basis.lower() in ['constant']:
        _basis = basisfunctionbem3d.CONSTANT
    elif basis.lower() in ['linear']:
        _basis = basisfunctionbem3d.LINEAR
    else:
        raise TypeError

    bem = new_slp_helmholtz_bem3d(k, grid._mesh.surface3d, q_reg, q_sing,
                                  _basis, _basis)
    root = build_bem3d_cluster(bem, clf, _basis)

    if strict:
        broot = build_strict_block(root, root, eta, admis)
    else:
        broot = build_nonstrict_block(root, root, eta, admis)

    if aprx.lower() in ['aca']:
        setup_hmatrix_aprx_inter_row_bem3d(bem, root, root, broot, m)
    elif aprx.lower() in ['paca']:
        setup_hmatrix_aprx_paca_bem3d(bem, root, root, broot, eps_aca)
    elif aprx.lower() in ['hca']:
        setup_hmatrix_aprx_hca_bem3d(bem, root, root, broot, m, eps_aca)
    elif aprx.lower() in ['inter_row']:
        setup_hmatrix_aprx_inter_row_bem3d(bem, root, root, broot, m)

    mat = build_from_block_hmatrix(broot, rk)
    start = timer()
    assemble_bem3d_hmatrix(bem, broot, mat)
    time_assemble = timer() - start

    Z = H2HMatrix(mat, root, broot)
    Z._time_assemble = time_assemble

    # keep references to h2lib objects so they don't get garbage collected
    # Z._root = root
    # important! don't ref bem and broot otherwise processes fail to terminate (not sure why)
    # Z._bem = bem
    # Z._broot = broot

    return Z


@numba.njit(cache=True)
def gauss_quadrature(n, type=1):
    '''
    Gaussian quadrature rules for triangular element surface integrals.
    '''
    if n == 1:
        return [[1 / 3, 1 / 3]], [
            1,
        ]
    elif n == 2:
        if type == 1:
            return [[1 / 6, 1 / 6], [2 / 3, 1 / 6],
                    [1 / 6, 2 / 3]], [1 / 3, 1 / 3, 1 / 3]
        elif type == 2:
            return [[0, 1 / 2], [1 / 2, 0], [1 / 2,
                                             1 / 2]], [1 / 3, 1 / 3, 1 / 3]
    elif n == 3:
        if type == 1:
            return [[1 / 3, 1 / 3], [1 / 5, 3 / 5], [1 / 5, 1 / 5],
                    [3 / 5, 1 / 5]],
            [-27 / 48, 25 / 48, 25 / 48, 25 / 48]
        elif type == 2:
            return [[1 / 3, 1 / 3], [2 / 15, 11 / 15], [2 / 15, 2 / 15],
                    [11 / 15, 2 / 15]],
            [-27 / 48, 25 / 48, 25 / 48, 25 / 48]


@numba.njit(cache=True)
def kernel_helmholtz(k, x1, y1, z1, x2, y2, z2):
    '''
    Helmholtz kernel for acoustic waves.
    '''
    r = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
    return math.exp(-1j * k * r) / (4 * np.pi * r)


@numba.jit(cache=True)
def pf_vec_np(grid, r, k, c, rho, gn=2):
    '''
    Frequency-domain pressure calculation vector from surface mesh.
    '''
    nodes = grid.vertices
    triangles = grid.triangles
    triangle_areas = grid.triangle_areas

    p = np.zeros(len(nodes), dtype=np.complex128)
    x, y, z = r
    gr, gw = gauss_quadrature(gn)

    for tt in range(len(triangles)):
        tri = triangles[tt, :]
        x1, y1 = nodes[tri[0], :2]
        x2, y2 = nodes[tri[1], :2]
        x3, y3 = nodes[tri[2], :2]

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


def pfir_from_layout(layout, grid, db_file, r, c, rho, use_kkr=False, interp=2):
    '''
    Pressure spatial impulse response for an array patch.
    '''
    # read database
    freqs, pnfr, nodes = database.read_patch_to_node_freq_resp(db_file)

    nctrldom = pnfr.shape[0]

    sfr = np.zeros((nctrldom, len(freqs)), dtype=np.complex128)

    for i, f in enumerate(freqs):
        omg = 2 * np.pi * f
        k = omg / c

        p_vector = pf_vec_np(grid, r, k, c, rho)

        for j in range(nctrldom):

            disp = pnfr[j, :, i]
            sfr[j, i] = p_vector.dot(disp)

    sir_t, sir = impulse_response.fft_to_sir(freqs,
                                             sfr,
                                             interp=interp,
                                             axis=1,
                                             use_kkr=use_kkr)

    return sir_t, sir


# def press_resp(array, refn, db_file, pes, r, c, rho, use_kkr=False, interp=2):
#     '''
#     Calculates the field pressure from an array.
#     '''
#     sir_t, sir = array_patch_pres_imp_resp(array,
#                                            refn,
#                                            db_file,
#                                            r,
#                                            c,
#                                            rho,
#                                            interp=interp,
#                                            use_kkr=use_kkr)
#     sir = sir.T

#     dt = sir_t[1] - sir_t[0]

#     ppatch = []
#     for i in range(pes.shape[1]):
#         ppatch.append(np.convolve(pes[:, i], sir[:, i]) * dt)

#     return np.sum(ppatch, axis=0)
