'''Routines for the boundary element method.'''
__all__ = [
    'z_mat_np', 'z_mat_fm', 'z_mat_hm', 'z_mat_hm_from_grid',
    'z_mat_fm_from_grid', 'z_mat_np_from_grid'
]
import numpy as np
from timeit import default_timer as timer
from cnld.matrix import H2FullMatrix, H2HMatrix
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
