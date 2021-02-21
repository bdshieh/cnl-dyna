''''''
__all__ = ['m_mat_np']

import numpy as np
import numba


def m_mat_np(grid, geom, mu=0.5):
    '''
    Mass matrix based on average of lumped and consistent mass matrix
    (lumped-consistent).
    '''
    DLM = mem_dlm_matrix(grid, geom)
    CMM = mem_cm_matrix(mem, geom)

    return mu * DLM + (1 - mu) * CMM


def m_con_mat_np(grid, geom):
    '''
    Mass matrix based on kinetic energy and linear shape functions
    (consistent).
    '''
    # get mesh information
    nodes = grid.vertices
    triangles = grid.triangles
    triangle_areas = grid.triangle_areas

    mass = geom.density * geom.thickness

    # construct M matrix by adding contribution from each element
    M = np.zeros((len(nodes), len(nodes)))
    for tt in range(len(triangles)):
        tri = triangles[tt, :]
        xi, yi = nodes[tri[0], :2]
        xj, yj = nodes[tri[1], :2]
        xk, yk = nodes[tri[2], :2]

        # da = ((xj - x5) * (yk - yi) - (xk - x5) * (yj - yi))
        da = triangle_areas[tt]
        Mt = np.array([[1, 1 / 2, 1 / 2], [1 / 2, 1, 1 / 2], [1 / 2, 1 / 2, 1]
                       ]) / 12
        M[np.ix_(tri, tri)] += 2 * Mt * mass * da

    ob = grid.on_boundary
    M[ob, :] = 0
    M[:, ob] = 0
    M[ob, ob] = 1

    return M


def m_dl_mat_np(grid, geom):
    '''
    Mass matrix based on equal distribution of element mass to nodes
    (diagonally-lumped).
    '''
    # get mesh information
    nodes = grid.vertices
    triangles = grid.triangles
    triangle_areas = grid.triangle_areas

    mass = mem.density * mem.thickness

    # construct M matrix by adding contribution from each element
    M = np.zeros((len(nodes), len(nodes)))
    for tt in range(len(triangles)):
        tri = triangles[tt, :]
        ap = triangle_areas[tt]
        M[tri, tri] += 1 / 3 * mass * ap

    ob = grid.on_boundary
    M[ob, :] = 0
    M[:, ob] = 0
    M[ob, ob] = 1

    return M