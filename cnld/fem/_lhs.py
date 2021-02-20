''''''
import numpy as np
import numpy.linalg
from cnld import abstract, mesh, util
from scipy import linalg
from scipy import sparse as sps
from scipy.constants import epsilon_0 as e_0
from scipy.integrate import dblquad
import numba


def inv_block(a):
    '''
    Inverse of a dense block matrix.
    '''
    return np.linalg.inv(a)


def mbk_mat_sps_from_layout(layout, grids, f, inv=False):
    '''
    Mass, Damping, and Stiffness matrix in sparse format for an array.
    '''
    omg = 2 * np.pi * f

    mapping = layout.membrane_to_geometry_mapping
    if mapping is None:
        gid = cycle(range(len(layout.geometries)))
        mapping = [next(gid) for i in range(len(layout.membranes))]

    mbk_list = [None] * len(layout.geometries)
    if inv:
        mbk_inv_list = [None] * len(layout.geometries)

    for i, geom in enumerate(layout.geometries):

        M = m_mat_np(grids.fem[i], geom)
        K = k_mat_np(grids.fem[i], geom)
        B = b_eig_mat_np(grids.fem[i], geom, M, K)
        mbk = -(omg**2) * M - 1j * omg * B + K

        mbk_list[i] = mbk
        if inv:
            mbk_inv_list[i] = inv_block(mbk)

    blocks = [None] * len(layout.membranes)
    if inv:
        blocks_inv = [None] * len(layout.membranes)

    for i, mem in layout.membranes:

        blocks[i] = mbk_list[mapping[i]]
        if inv:
            blocks_inv[i] = mbk_inv_list[mapping[i]]
            
    if inv:
        return sps.block_diag(blocks,
                              format='csr'), sps.block_diag(blocks_inv,
                                                            format='csr')
    else:
        return sps.block_diag(blocks, format='csr')
