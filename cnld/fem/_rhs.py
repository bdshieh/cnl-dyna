import numpy as np
import numpy.linalg
from cnld import abstract, mesh, util
from scipy import linalg
from scipy import sparse as sps
from scipy.constants import epsilon_0 as e_0
from scipy.integrate import dblquad
import numba

eps = np.finfo(np.float64).eps

# np
# sps
# hm
# am
# spm

def p_vec_np(grid, p):
    '''
    Pressure load vector based on equal distribution of pressure to element
    nodes.
    '''
    nodes = grid.vertices
    triangles = grid.triangles
    triangle_areas = grid.triangle_areas
    ob = grid.on_boundary

    f = np.zeros(len(nodes))
    for tt in range(len(triangles)):
        tri = triangles[tt, :]
        ap = triangle_areas[tt]
        bfac = 1 * np.sum(~ob[tri])

        f[tri] += 1 / bfac * p * ap

    f[ob] = 0

    return f


def p_arb_vec_np(grid, pfun):
    '''
    Pressure load vector based on an arbitrary load.
    '''
    nodes = grid.vertices
    triangles = grid.triangles
    triangle_areas = grid.triangle_areas
    ob = grid.on_boundary

    f = np.zeros(len(nodes))
    for tt in range(len(triangles)):
        tri = triangles[tt, :]
        xi, yi = nodes[tri[0], :2]
        xj, yj = nodes[tri[1], :2]
        xk, yk = nodes[tri[2], :2]

        def pfun_psi_eta(psi, eta):
            x = (xj - xi) * psi + (xk - xi) * eta + xi
            y = (yj - yi) * psi + (yk - yi) * eta + yi
            return pfun(x, y)

        integ, _ = dblquad(pfun_psi_eta,
                           0,
                           1,
                           0,
                           lambda x: 1 - x,
                           epsrel=1e-1,
                           epsabs=1e-1)
        frac = integ / (1 / 2)  # fraction of triangle covered by load
        da = triangle_areas[tt]
        bfac = 1 * np.sum(~ob[tri])

        f[tri] += 1 / bfac * frac * da

    ob = amesh.on_boundary
    f[ob] = 0

    return f


def p_cd_mat_np(grid, geom):
    '''
    Load vector for a patch.
    '''
    nodes = grid.vertices
    triangles = grid.triangles
    triangle_areas = grid.triangle_areas
    ob = grid.on_boundary

    ctrldomlist = geometry_to_controldomainlist(geom)

    f = []
    for pat in mem.patches:

        if isinstance(mem, abstract.SquareCmutMembrane):

            px, py, pz = pat.position
            plx = pat.length_x
            ply = pat.length_y

            def load_func(x, y):
                # use 2 * eps to account for potential round-off error
                if x - (px - plx / 2) >= -2 * eps:
                    if x - (px + plx / 2) <= 2 * eps:
                        if y - (py - ply / 2) >= -2 * eps:
                            if y - (py + ply / 2) <= 2 * eps:
                                return 1
                return 0
        else:
            px, py, pz = pat.position
            prmin = pat.radius_min
            prmax = pat.radius_max
            pthmin = pat.theta_min
            pthmax = pat.theta_max

            def load_func(x, y):
                r = np.sqrt((x - px)**2 + (y - py)**2)
                th = np.arctan2((y - py), (x - px))
                # pertube theta by 2 * eps to account for potential round-off error
                th1 = th - 2 * eps
                if th1 < -np.pi:
                    th1 += 2 * np.pi  # account for [-pi, pi] wrap-around
                th2 = th + 2 * eps
                if th2 > np.pi:
                    th2 -= 2 * np.pi  # account for [-pi, pi] wrap-around
                if r - prmin >= -2 * eps:
                    if r - prmax <= 2 * eps:
                        # for theta, check both perturbed values
                        if th1 - pthmin >= 0:
                            if th1 - pthmax <= 0:
                                return 1
                        if th2 - pthmin >= 0:
                            if th2 - pthmax <= 0:
                                return 1
                return 0

        f_pat = np.zeros(len(nodes))
        for tt in range(len(triangles)):
            tri = triangles[tt, :]
            xi, yi = nodes[tri[0], :2]
            xj, yj = nodes[tri[1], :2]
            xk, yk = nodes[tri[2], :2]

            # check if triangle vertices are inside or outside load
            loadi = load_func(xi, yi)
            loadj = load_func(xj, yj)
            loadk = load_func(xk, yk)
            # if load covers entire triangle
            if all([loadi, loadj, loadk]):
                da = triangle_areas[tt]
                bfac = 1 * np.sum(~ob[tri])

                f_pat[tri] += 1 / bfac * 1 * da
            # if load does not cover any part of triangle
            elif not any([loadi, loadj, loadk]):
                continue
            # if load partially covers triangle
            else:

                def load_func_psi_eta(psi, eta):
                    x = (xj - xi) * psi + (xk - xi) * eta + xi
                    y = (yj - yi) * psi + (yk - yi) * eta + yi
                    return load_func(x, y)

                integ, _ = dblquad(load_func_psi_eta,
                                   0,
                                   1,
                                   0,
                                   lambda x: 1 - x,
                                   epsrel=1e-1,
                                   epsabs=1e-1)
                frac = integ / (1 / 2)  # fraction of triangle covered by load
                da = triangle_areas[tt]
                bfac = 1 * np.sum(~ob[tri])

                f_pat[tri] += 1 / bfac * frac * da

        f_pat[ob] = 0
        f.append(f_pat)

    return np.array(f).T


def avg_cd_mat_np(grid, geom):
    '''
    Averaging vector for a patch.
    '''


    nodes = amesh.vertices
    triangles = amesh.triangles
    triangle_areas = amesh.triangle_areas
    # ob = amesh.on_boundary

    avg = []
    for pat in mem.patches:
        if isinstance(mem, abstract.SquareCmutMembrane):

            px, py, pz = pat.position
            plx = pat.length_x
            ply = pat.length_y

            def load_func(x, y):
                # use 2 * eps to account for potential round-off error
                if x - (px - plx / 2) >= -2 * eps:
                    if x - (px + plx / 2) <= 2 * eps:
                        if y - (py - ply / 2) >= -2 * eps:
                            if y - (py + ply / 2) <= 2 * eps:
                                return 1
                return 0
        else:
            px, py, pz = pat.position
            prmin = pat.radius_min
            prmax = pat.radius_max
            pthmin = pat.theta_min
            pthmax = pat.theta_max

            def load_func(x, y):
                r = np.sqrt((x - px)**2 + (y - py)**2)
                th = np.arctan2((y - py), (x - px))
                # pertube theta by 2 * eps to account for potential round-off error
                th1 = th - 2 * eps
                if th1 < -np.pi:
                    th1 += 2 * np.pi  # account for [-pi, pi] wrap-around
                th2 = th + 2 * eps
                if th2 > np.pi:
                    th2 -= 2 * np.pi  # account for [-pi, pi] wrap-around
                if r - prmin >= -2 * eps:
                    if r - prmax <= 2 * eps:
                        # for theta, check both perturbed values
                        if th1 - pthmin >= 0:
                            if th1 - pthmax <= 0:
                                return 1
                        if th2 - pthmin >= 0:
                            if th2 - pthmax <= 0:
                                return 1
                return 0

        avg_pat = np.zeros(len(nodes))
        for tt in range(len(triangles)):
            tri = triangles[tt, :]
            xi, yi = nodes[tri[0], :2]
            xj, yj = nodes[tri[1], :2]
            xk, yk = nodes[tri[2], :2]

            # check if triangle vertices are inside or outside load
            loadi = load_func(xi, yi)
            loadj = load_func(xj, yj)
            loadk = load_func(xk, yk)
            # if load covers entire triangle
            if all([loadi, loadj, loadk]):
                da = triangle_areas[tt]

                avg_pat[tri] += 1 / 3 * 1 * da
            # if load does not cover any part of triangle
            elif not any([loadi, loadj, loadk]):
                continue
            # if load partially covers triangle
            else:

                def load_func_psi_eta(psi, eta):
                    x = (xj - xi) * psi + (xk - xi) * eta + xi
                    y = (yj - yi) * psi + (yk - yi) * eta + yi
                    return load_func(x, y)

                integ, _ = dblquad(load_func_psi_eta,
                                   0,
                                   1,
                                   0,
                                   lambda x: 1 - x,
                                   epsrel=1e-1,
                                   epsabs=1e-1)
                frac = integ / (1 / 2)  # fraction of triangle covered by load
                da = triangle_areas[tt]

                avg_pat[tri] += 1 / 3 * frac * da

        avg.append(avg_pat / pat.area)

    return np.array(avg).T


# def mem_patch_fcol_vector(mem, refn):
#     '''
#     '''
#     f = mem_patch_f_matrix(mem, refn)
#     avg = mem_patch_avg_matrix(mem, refn)

#     K = mem_k_matrix(mem, refn)
#     Kinv = inv_block(K)
#     u = Kinv.dot(-np.sum(f, axis=1)).squeeze()
#     u = u / np.max(np.abs(u))

#     g = mem.gap

#     fcol = []
#     ups = []

#     for i, pat in enumerate(mem.patches):

#         # f_pat = f[:,i]
#         # u = Kinv.dot(-f_pat).squeeze()
#         # u = u / np.max(np.abs(u))
#         avg_pat = avg[:, i]
#         scale = -g / u[avg_pat > 0].min()
#         # scale = -g / u.dot(avg_pat)
#         up = scale * u
#         up[up < -g] = -g

#         p = (K.dot(up)).dot(avg_pat) / pat.area
#         fcol.append(-p)
#         ups.append(up)
#         # fcol.append(-g / u[f_pat > 0].min())

#     return np.array(fcol), np.array(ups)


def p_mat_sps_from_layout(layout, grids):
    '''
    Construct load vector based on patches of abstract array.
    '''

    mapping = layout.membrane_to_geometry_mapping
    if mapping is None:
        gid = cycle(range(len(layout.geometries)))
        mapping = [next(gid) for i in range(len(layout.membranes))]

    p_list = [None] * len(layout.geometries)
    for i, geom, grid in enumarte(layout.geometries):
        p_list[i] = p_cd_mat_np(grids.fem[i], geom)

    blocks = [None] * len(layout.membranes)

    for i, mem in enumerate(layout.membranes):
        blocks[i] = p_list[mapping[i]]

    return sps.block_diag(blocks, format='csc')


def avg_mat_sps_from_layout(layout, grids):
    '''
    Construct load vector based on patches of abstract array.
    '''
    mapping = layout.membrane_to_geometry_mapping
    if mapping is None:
        gid = cycle(range(len(layout.geometries)))
        mapping = [next(gid) for i in range(len(layout.membranes))]

    avg_list = [None] * len(layout.geometries)
    for i, geom, grid in enumarte(layout.geometries):
        avg_list[i] = avg_cd_mat_np(grids.fem[i], geom)

    blocks = [None] * len(layout.membranes)

    for i, mem in enumerate(layout.membranes):
        blocks[i] = avg_list[mapping[i]]

    return sps.block_diag(blocks, format='csc')