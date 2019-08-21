'''
'''
import numpy as np
import numpy.linalg
from scipy import sparse as sps, linalg
from scipy.integrate import dblquad
from scipy.constants import epsilon_0 as e_0

from cnld import util, abstract, mesh


eps = np.finfo(np.float64).eps


@util.memoize
def mem_static_x_vector(mem, refn, vdc, atol=1e-10, maxiter=100):
    '''
    '''
    def pes(v, x, g_eff):
        return -e_0 / 2 * v ** 2 / (g_eff + x) ** 2

    if isinstance(mem, abstract.SquareCmutMembrane):
        amesh = mesh.square(mem.length_x, mem.length_y, refn)
    else:
        amesh = mesh.circle(mem.radius, refn)

    K = mem_k_matrix(mem, refn)
    g_eff = mem.gap + mem.isolation / mem.permittivity
    F = mem_f_vector(mem, refn, 1)
    Kinv = linalg.inv(K)
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


@util.memoize
def mem_k_matrix(mem, refn):
    '''
    Stiffness matrix based on 3-dof (rotation-free) triangular plate elements.
    Refer to E. Onate and F. Zarate, Int. J. Numer. Meth. Engng. 47, 557-603 (2000).
    '''
    def L(x1, y1, x2, y2):
        # calculates edge length
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def T_matrix(x1, y1, x2, y2):
        # transformation matrix for an edge
        z = [0, 0, 1]
        r = [x2 - x1, y2 - y1]
        n = np.cross(z, r)
        norm = np.linalg.norm(n)
        n = n / norm
        nx, ny, _ = n
        return np.array([[-nx, 0], [0, -ny], [-ny, -nx]])

    if isinstance(mem, abstract.SquareCmutMembrane):
        amesh = mesh.square(mem.length_x, mem.length_y, refn)
    else:
        amesh = mesh.circle(mem.radius, refn)

    # get mesh information
    nodes = amesh.vertices
    triangles = amesh.triangles
    triangle_edges = amesh.triangle_edges
    triangle_areas = amesh.triangle_areas
    ntriangles = len(triangles)
    ob = amesh.on_boundary

    # determine list of neighbors for each triangle
    # None indicates neighbor doesn't exist for that edge (boundary edge)
    triangle_neighbors = []
    for tt in range(ntriangles):
        neighbors = []
        for te in triangle_edges[tt, :]:
            mask = np.any(triangle_edges == te, axis=1)
            args = np.nonzero(mask)[0]
            if len(args) > 1:
                neighbors.append(args[args != tt][0])
            else:
                neighbors.append(None)
        triangle_neighbors.append(neighbors)
    amesh.triangle_neighbors = triangle_neighbors

    # construct constitutive matrix for material
    h = mem.thickness[0]  # no support for composite membranes yet
    E = mem.y_modulus[0]
    eta = mem.p_ratio[0]

    D = np.zeros((3, 3))
    D[0, 0] = 1
    D[0, 1] = eta
    D[1, 0] = eta
    D[1, 1] = 1
    D[2, 2] = (1 - eta) / 2
    D = D * E * h**3 / (12 * (1 - eta**2))

    # calculate Jacobian and gradient operator for each triangle
    gradops = []
    for tt in range(ntriangles):
        tri = triangles[tt, :]
        xi, yi = nodes[tri[0], :2]
        xj, yj = nodes[tri[1], :2]
        xk, yk = nodes[tri[2], :2]

        J = np.array([[xj - xi, xk - xi], [yj - yi, yk - yi]])
        gradop = np.linalg.inv(J.T).dot([[-1, 1, 0], [-1, 0, 1]])
        # gradop = np.linalg.inv(J.T).dot([[1, 0, -1],[0, 1, -1]])

        gradops.append(gradop)

    # construct K matrix
    K = np.zeros((len(nodes), len(nodes)))
    for p in range(ntriangles):
        trip = triangles[p, :]
        ap = triangle_areas[p]

        xi, yi = nodes[trip[0], :2]
        xj, yj = nodes[trip[1], :2]
        xk, yk = nodes[trip[2], :2]

        neighbors = triangle_neighbors[p]
        gradp = gradops[p]
        # list triangle edges, ordered so that z cross-product will produce outward normal
        edges = [(xk, yk, xj, yj), (xi, yi, xk, yk), (xj, yj, xi, yi)]

        # begin putting together indexes needed later for matrix assignment
        ii, jj, kk = trip
        Kidx = [ii, jj, kk]
        Kpidx = [0, 1, 2]

        # construct B matrix for control element
        Bp = np.zeros((3, 6))
        for j, n in enumerate(neighbors):
            if n is None:
                continue

            # determine index of the node in the neighbor opposite edge
            iin, jjn, kkn = triangles[n, :]
            uidx = [x for x in np.unique([ii, jj, kk, iin, jjn, kkn]) if x not in [
                ii, jj, kk]][0]
            # update indexes
            Kidx.append(uidx)
            Kpidx.append(3 + j)

            l = L(*edges[j])
            T = T_matrix(*edges[j])
            gradn = gradops[n]

            pterm = l / 2 * T.dot(gradp)
            Bp[:, :3] += pterm

            nterm = l / 2 * T.dot(gradn)
            idx = [Kpidx[Kidx.index(x)] for x in [iin, jjn, kkn]]
            Bp[:, idx] += nterm
        Bp = Bp / ap

        # construct local K matrix for control element
        Kp = (Bp.T).dot(D).dot(Bp) * ap

        # add matrix values to global K matrix
        K[np.ix_(Kidx, Kidx)] += Kp[np.ix_(Kpidx, Kpidx)]

    K[ob, :] = 0
    K[:, ob] = 0
    K[ob, ob] = 1

    return K


@util.memoize
def mem_k_matrix_hpb(mem, refn):
    '''
    Stiffness matrix based on 3-dof (rotation-free) triangular plate elements.
    '''
    def mag(r1, r2):
        # calculates edge length
        return np.sqrt((r2[0] - r1[0])**2 + (r2[1] - r1[1])**2)

    if isinstance(mem, abstract.SquareCmutMembrane):
        amesh = mesh.square(mem.length_x, mem.length_y, refn)
    else:
        amesh = mesh.circle(mem.radius, refn)

    # get mesh information
    nodes = amesh.vertices
    triangles = amesh.triangles
    triangle_edges = amesh.triangle_edges
    triangle_areas = amesh.triangle_areas
    ntriangles = len(triangles)
    ob = amesh.on_boundary

    # determine list of neighbors for each triangle
    # None indicates neighbor doesn't exist for that edge (boundary edge)
    triangle_neighbors = []
    for tt in range(ntriangles):
        neighbors = []
        for te in triangle_edges[tt, :]:
            mask = np.any(triangle_edges == te, axis=1)
            args = np.nonzero(mask)[0]
            if len(args) > 1:
                neighbors.append(args[args != tt][0])
            else:
                neighbors.append(None)

                # determine index of the node in the neighbor opposite edge
                iin, jjn, kkn = triangles[neighbors[0], :]
                uidx = [x for x in np.unique([ii, jj, kk, iin, jjn, kkn]) if x not in [
                    ii, jj, kk]][0]
                # update indexes
                Kidx.append(uidx)
                Kpidx.append(3 + j)

        triangle_neighbors.append(neighbors)
    amesh.triangle_neighbors = triangle_neighbors

    # construct constitutive matrix for material
    h = mem.thickness[0]  # no support for composite membranes yet
    E = mem.y_modulus[0]
    eta = mem.p_ratio[0]

    D = np.zeros((3, 3))
    D[0, 0] = 1
    D[0, 1] = eta
    D[1, 0] = eta
    D[1, 1] = 1
    D[2, 2] = (1 - eta)
    D = D * E * h**3 / (12 * (1 - eta**2))

    # construct K matrix
    K = np.zeros((len(nodes), len(nodes)))
    for p in range(ntriangles):
        trip = triangles[p, :]
        ap = triangle_areas[p]

        x5, y5 = nodes[trip[0], :2]
        x6, y6 = nodes[trip[1], :2]
        x4, y4 = nodes[trip[2], :2]

        r4 = np.array([x4, y4])
        r5 = np.array([x5, y5])
        r6 = np.array([x6, y6])

        neighbors = triangle_neighbors[p]

        if neighbors[0] is None:
            x = r5
            xo = r6
            n = (r4 - r6) / mag(r4, r6)
            x1, y1 = -x + 2 * xo + (x - xo).dot(n) * n
        else:
            x1, y1 = triangles[neighbors[0], :2]

        if neighbors[1] is None:
            x = r6
            xo = r4
            n = (r5 - r4) / mag(r5, r4)
            x2, y2 = -x + 2 * xo + (x - xo).dot(n) * n
        else:
            x2, y2 = triangles[neighbors[1], :2]

        if neighbors[2] is None:
            x = r4
            xo = r5
            n = (r6 - r5) / mag(r6, r5)
            x3, y3 = -x + 2 * xo + (x - xo).dot(n) * n
        else:
            x3, y3 = triangles[neighbors[2], :2]

        r1 = np.array([x1, y1])
        r2 = np.array([x2, y2])
        r3 = np.array([x3, y3])

        C = np.zeros((6, 3))
        C[0, :] = [x1**2 / 2, y1**2 / 2, x1 * y1]
        C[1, :] = [x2**2 / 2, y2**2 / 2, x2 * y2]
        C[2, :] = [x3**2 / 2, y3**2 / 2, x3 * y3]
        C[3, :] = [x4**2 / 2, y4**2 / 2, x4 * y4]
        C[4, :] = [x5**2 / 2, y5**2 / 2, x5 * y5]
        C[5, :] = [x6**2 / 2, y6**2 / 2, x6 * y6]

        # L1 = np.sqrt((x6 - x4)**2 + (y6 - y4)**2)
        # b1a1 = ((x1 - x4) * (x6 - x4) + (y1 - y4) * (y6 - y4)) / np.sqrt((x6 - x4)**2 + (y6 - y4)**2)
        # b1a2 = ((x1 - x6) * (x4 - x6) + (y1 - y6) * (y4 - y6)) / np.sqrt((x4 - x6)**2 + (y4 - y6)**2)
        # b1b1 = ((x5 - x4) * (x6 - x4) + (y5 - y4) * (y6 - y4)) / np.sqrt((x6 - x4)**2 + (y6 - y4)**2)
        # b1b2 = ((x5 - x6) * (x4 - x6) + (y5 - y6) * (y4 - y6)) / np.sqrt((x4 - x6)**2 + (y4 - y6)**2)
        # h1a = np.sqrt(np.sqrt((x1 - x4)**2 + (y1 - y4)**2) - b1a1**2)
        # h1b = np.sqrt(np.sqrt((x5 - x6)**2 + (y5 - y6)**2) - b1b2**2)

        L1 = mag(r6, r4)
        b1a1 = (r1 - r4).dot(r6 - r4) / L1
        b1a2 = (r1 - r6).dot(r4 - r6) / L1
        b1b1 = (r5 - r4).dot(r6 - r4) / L1
        b1b2 = (r5 - r6).dot(r4 - r6) / L1
        h1a = np.sqrt(mag(r1, r4)**2 - b1a1**2)
        h1b = np.sqrt(mag(r5, r6)**2 - b1b2**2)

        L2 = mag(r5, r4)
        b2a1 = (r2 - r5).dot(r4 - r5) / L2
        b2a2 = (r2 - r4).dot(r5 - r4) / L2
        b2b1 = (r6 - r5).dot(r4 - r5) / L2
        b2b2 = (r6 - r4).dot(r4 - r5) / L2
        h2a = np.sqrt(mag(r2, r5)**2 - b2a1**2)
        h2b = np.sqrt(mag(r6, r4)**2 - b2b2**2)

        L3 = mag(r5, r6)
        b3a1 = (r3 - r6).dot(r5 - r6) / L3
        b3a2 = (r3 - r5).dot(r6 - r5) / L3
        b3b1 = (r4 - r6).dot(r5 - r6) / L3
        b3b2 = (r4 - r5).dot(r6 - r5) / L3
        h3a = np.sqrt(mag(r3, r6)**2 - b3a1**2)
        h3b = np.sqrt(mag(r4, r5)**2 - b3b2**2)

        L = np.zeros((3, 6))
        L[0, 0] = 1 / h1a
        L[1, 1] = 1 / h2a
        L[2, 2] = 1 / h3a
        L[0, 3] = -b1a2 / (L1 * h1a) + -b1b2 / (L1 * h1b)
        L[0, 4] = 1 / h1b
        L[0, 5] = -b1a1 / (L1 * h1a) + -b1b1 / (L1 * h1b)
        L[1, 3] = -b2b1 / (L2 * h2b) + -b2a1 / (L2 * h2a)
        L[1, 4] = -b2b2 / (L2 * h2b) + -b2a2 / (L2 * h2a)
        L[1, 5] = 1 / h2b
        L[2, 3] = 1 / h3b
        L[2, 4] = -b3a1 / (L3 * h3a) + -b3b1 / (L3 * h3b)
        L[2, 5] = -b3b2 / (L3 * h3b) + -b3a2 / (L3 * h3a)

        G = L.dot(C)
        Ginv = np.linalg.inv(G)

        I_D = np.zeros((3, 3))
        I_D[0, 0] = 1
        I_D[1, 1] = 1
        I_D[2, 2] = 2

        K_be = ap * (L.T).dot(Ginv.T).dot(I_D).dot(D).dot(Ginv).dot(L)

        # begin putting together indexes needed later for matrix assignment
        K_idx = [trip[2], trip[0], trip[1]]
        K_be_idx = [4, 5, 6]

        # apply BCs
        if neighbors[0] is None:

            K_be[3, 3] /= 2
            K_be[4, 3] = (K_be[4, 3] + K_be[0, 3]) / 2
            K_be[3, 4] = (K_be[3, 4] + K_be[3, 0]) / 2

            K_be[5, 5] /= 2
            K_be[4, 5] = (K_be[4, 5] + K_be[0, 5]) / 2
            K_be[5, 4] = (K_be[5, 4] + K_be[5, 0]) / 2

            K_be[4, 4] = (K_be[0, 0] + K_be[0, 4] + K_be[4, 0] + K_be[4, 4]) / 2

        else:

            K_idx.append(neighbors[0])
            K_be_idx.append(0)
        
        if neighbors[1] is None:

            K_be[3, 3] /= 2
            K_be[5, 3] = (K_be[5, 3] + K_be[1, 3]) / 2
            K_be[3, 5] = (K_be[3, 5] + K_be[3, 1]) / 2

            K_be[4, 4] /= 2
            K_be[5, 4] = (K_be[5, 4] + K_be[1, 4]) / 2
            K_be[4, 5] = (K_be[4, 5] + K_be[4, 1]) / 2

            K_be[5, 5] = (K_be[1, 1] + K_be[1, 5] + K_be[5, 1] + K_be[5, 5]) / 2

        else:
            K_idx.append(neighbors[1])
            K_be_idx.append(1)

        if neighbors[2] is None:

            K_be[4, 4] /= 2
            K_be[3, 4] = (K_be[3, 4] + K_be[2, 4]) / 2
            K_be[4, 3] = (K_be[4, 3] + K_be[4, 2]) / 2

            K_be[5, 5] /= 2
            K_be[5, 3] = (K_be[5, 3] + K_be[2, 5]) / 2
            K_be[3, 5] = (K_be[3, 5] + K_be[5, 2]) / 2

            K_be[3, 3] = (K_be[2, 2] + K_be[2, 3] + K_be[3, 2] + K_be[3, 3]) / 2

        else:
            K_idx.append(neighbors[2])
            K_be_idx.append(2)

        # add matrix values to global K matrix
        K[np.ix_(K_idx, K_idx)] += K_be[np.ix_(K_be_idx, K_be_idx)]

    K[ob, :] = 0
    K[:, ob] = 0
    K[ob, ob] = 1

    return K


@util.memoize
def mem_m_matrix(mem, refn, mu=0.5):
    '''
    Mass matrix based on average of lumped and consistent mass matrix (lumped-consistent).
    '''
    DLM = mem_dlm_matrix(mem, refn)
    CMM = mem_cm_matrix(mem, refn)

    return mu * DLM + (1 - mu) * CMM


@util.memoize
def mem_cm_matrix(mem, refn):
    '''
    Mass matrix based on kinetic energy and linear shape functions (consistent).
    '''
    if isinstance(mem, abstract.SquareCmutMembrane):
        amesh = mesh.square(mem.length_x, mem.length_y, refn)
    else:
        amesh = mesh.circle(mem.radius, refn)

    # get mesh information
    nodes = amesh.vertices
    triangles = amesh.triangles
    triangle_areas = amesh.triangle_areas

    mass = sum([x * y for x, y in zip(mem.density, mem.thickness)])

    # construct M matrix by adding contribution from each element
    M = np.zeros((len(nodes), len(nodes)))
    for tt in range(len(triangles)):
        tri = triangles[tt, :]
        xi, yi = nodes[tri[0], :2]
        xj, yj = nodes[tri[1], :2]
        xk, yk = nodes[tri[2], :2]

        # da = ((xj - xi) * (yk - yi) - (xk - xi) * (yj - yi))
        da = triangle_areas[tt]
        Mt = np.array(
            [[1, 1 / 2, 1 / 2], [1 / 2, 1, 1 / 2], [1 / 2, 1 / 2, 1]]) / 12
        M[np.ix_(tri, tri)] += 2 * Mt * mass * da

    ob = amesh.on_boundary
    M[ob, :] = 0
    M[:, ob] = 0
    M[ob, ob] = 1

    return M


@util.memoize
def mem_dlm_matrix(mem, refn):
    '''
    Mass matrix based on equal distribution of element mass to nodes (diagonally-lumped).
    '''
    if isinstance(mem, abstract.SquareCmutMembrane):
        amesh = mesh.square(mem.length_x, mem.length_y, refn)
    else:
        amesh = mesh.circle(mem.radius, refn)

    # get mesh information
    nodes = amesh.vertices
    triangles = amesh.triangles
    triangle_areas = amesh.g / 2

    mass = sum([x * y for x, y in zip(mem.density, mem.thickness)])

    # construct M matrix by adding contribution from each element
    M = np.zeros((len(nodes), len(nodes)))
    for tt in range(len(triangles)):
        tri = triangles[tt, :]
        ap = triangle_areas[tt]
        M[tri, tri] += 1 / 3 * mass * ap

    ob = amesh.on_boundary
    M[ob, :] = 0
    M[:, ob] = 0
    M[ob, ob] = 1

    return M


@util.memoize
def mem_b_matrix(mem, M, K):
    '''
    Damping matrix based on Rayleigh damping for damping ratios at two frequencies.
    '''
    fa = mem.damping_freq_a
    fb = mem.damping_freq_b
    za = mem.damping_ratio_a
    zb = mem.damping_ratio_b

    omga = 2 * np.pi * fa
    omgb = 2 * np.pi * fb

    # solve for alpha and beta
    A = 1 / 2 * np.array([[1 / omga, omga], [1 / omgb, omgb]])
    alpha, beta = linalg.inv(A).dot([za, zb])

    return alpha * M + beta * K


@util.memoize
def mem_eig(mem, refn):
    '''
    Returns the eigenfrequency (in Hz) and eigenmodes of a membrane.
    '''
    if isinstance(mem, abstract.SquareCmutMembrane):
        amesh = mesh.square(mem.length_x, mem.length_y, refn)
    else:
        amesh = mesh.circle(mem.radius, refn)
    ob = amesh.on_boundary

    M = mem_m_matrix(mem, refn, mu=0.5)
    K = mem_k_matrix(mem, refn)
    w, v = linalg.eig(linalg.inv(M).dot(K)[np.ix_(~ob, ~ob)])

    idx = np.argsort(np.sqrt(np.abs(w)))
    eigf = np.sqrt(np.abs(w))[idx] / (2 * np.pi)
    eigv = v[:, idx]

    return eigf, eigv


@util.memoize
def mem_b_matrix_eig(mem, refn, M, K):
    '''
    Damping matrix based on Rayleigh damping for damping ratios at two modal frequencies.
    '''
    # if isinstance(mem, abstract.SquareCmutMembrane):
    #     amesh = mesh.square(mem.length_x, mem.length_y, refn)
    # else:
    #     amesh = mesh.circle(mem.radius, refn)
    # ob = amesh.on_boundary

    ma = mem.damping_mode_a
    mb = mem.damping_mode_b
    za = mem.damping_ratio_a
    zb = mem.damping_ratio_b

    # determine eigenfrequencies of membrane
    # w, v = linalg.eig(linalg.inv(M).dot(K)[np.ix_(~ob, ~ob)])
    # omg = np.sort(np.sqrt(np.abs(w)))
    # omga = omg[ma]
    # omgb = omg[mb]
    eigf, _ = mem_eig(mem, refn)
    omga = eigf[ma] * 2 * np.pi
    omgb = eigf[mb] * 2 * np.pi

    # solve for alpha and beta
    A = 1 / 2 * np.array([[1 / omga, omga], [1 / omgb, omgb]])
    alpha, beta = linalg.inv(A).dot([za, zb])

    return alpha * M + beta * K


@util.memoize
def mem_f_vector(mem, refn, p):
    '''
    Pressure load vector based on equal distribution of pressure to element nodes.
    '''
    if isinstance(mem, abstract.SquareCmutMembrane):
        amesh = mesh.square(mem.length_x, mem.length_y, refn)
    else:
        amesh = mesh.circle(mem.radius, refn)

    nodes = amesh.vertices
    triangles = amesh.triangles
    triangle_areas = amesh.g / 2
    ob = amesh.on_boundary

    f = np.zeros(len(nodes))
    for tt in range(len(triangles)):
        tri = triangles[tt, :]
        ap = triangle_areas[tt]
        bfac = 1 * np.sum(~ob[tri])

        f[tri] += 1 / bfac * p * ap

    f[ob] = 0

    return f


def mem_f_vector_arb_load(mem, refn, load_func):
    '''
    Pressure load vector based on an arbitrary load.
    '''
    if isinstance(mem, abstract.SquareCmutMembrane):
        amesh = mesh.square(mem.length_x, mem.length_y, refn)
    else:
        amesh = mesh.circle(mem.radius, refn)

    nodes = amesh.vertices
    triangles = amesh.triangles
    triangle_areas = amesh.triangle_areas
    ob = amesh.on_boundary

    f = np.zeros(len(nodes))
    for tt in range(len(triangles)):
        tri = triangles[tt, :]
        xi, yi = nodes[tri[0], :2]
        xj, yj = nodes[tri[1], :2]
        xk, yk = nodes[tri[2], :2]

        def load_func_psi_eta(psi, eta):
            x = (xj - xi) * psi + (xk - xi) * eta + xi
            y = (yj - yi) * psi + (yk - yi) * eta + yi
            return load_func(x, y)

        integ, _ = dblquad(load_func_psi_eta, 0, 1, 0,
                           lambda x: 1 - x, epsrel=1e-1, epsabs=1e-1)
        frac = integ / (1 / 2)  # fraction of triangle covered by load
        da = triangle_areas[tt]
        bfac = 1 * np.sum(~ob[tri])

        f[tri] += 1 / bfac * frac * da

    ob = amesh.on_boundary
    f[ob] = 0

    return f


@util.memoize
def mem_patch_f_matrix(mem, refn):
    '''
    Load vector for a patch.
    '''
    if isinstance(mem, abstract.SquareCmutMembrane):
        amesh = mesh.square(mem.length_x, mem.length_y,
                            refn, center=mem.position)
    else:
        amesh = mesh.circle(mem.radius, refn, center=mem.position)

    nodes = amesh.vertices
    triangles = amesh.triangles
    triangle_areas = amesh.triangle_areas
    ob = amesh.on_boundary

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

                integ, _ = dblquad(load_func_psi_eta, 0, 1, 0,
                                   lambda x: 1 - x, epsrel=1e-1, epsabs=1e-1)
                frac = integ / (1 / 2)  # fraction of triangle covered by load
                da = triangle_areas[tt]
                bfac = 1 * np.sum(~ob[tri])

                f_pat[tri] += 1 / bfac * frac * da

        f_pat[ob] = 0
        f.append(f_pat)

    return np.array(f).T


@util.memoize
def mem_patch_avg_matrix(mem, refn):
    '''
    Averaging vector for a patch.
    '''
    if isinstance(mem, abstract.SquareCmutMembrane):
        amesh = mesh.square(mem.length_x, mem.length_y,
                            refn, center=mem.position)
    else:
        amesh = mesh.circle(mem.radius, refn, center=mem.position)

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

                integ, _ = dblquad(load_func_psi_eta, 0, 1, 0,
                                   lambda x: 1 - x, epsrel=1e-1, epsabs=1e-1)
                frac = integ / (1 / 2)  # fraction of triangle covered by load
                da = triangle_areas[tt]

                avg_pat[tri] += 1 / 3 * frac * da

        avg.append(avg_pat / pat.area)

    return np.array(avg).T


def array_f_spmatrix(array, refn):
    '''
    Construct load vector based on patches of abstract array.
    '''
    blocks = []
    for elem in array.elements:
        for mem in elem.membranes:
            f = mem_patch_f_matrix(mem, refn)
            blocks.append(f)

    return sps.block_diag(blocks, format='csc')


def array_avg_spmatrix(array, refn):
    '''
    Construct load vector based on patches of abstract array.
    '''
    blocks = []
    for elem in array.elements:
        for mem in elem.membranes:
            avg = mem_patch_avg_matrix(mem, refn)
            blocks.append(avg)

    return sps.block_diag(blocks, format='csc')


@util.memoize
def inv_block(a):
    return np.linalg.inv(a)


def array_mbk_spmatrix(array, refn, f, inv=False):
    '''
    '''
    omg = 2 * np.pi * f
    blocks = []
    if inv:
        blocks_inv = []
    for elem in array.elements:
        for mem in elem.membranes:

            M = mem_m_matrix(mem, refn, mu=0.5)
            K = mem_k_matrix(mem, refn)
            # B = mem_b_matrix(mem, M, K)
            B = mem_b_matrix_eig(mem, refn, M, K)

            block = -(omg**2) * M - 1j * omg * B + K

            blocks.append(block)
            if inv:
                blocks_inv.append(inv_block(block))
    if inv:
        return sps.block_diag(blocks, format='csr'), sps.block_diag(blocks_inv, format='csr')
    else:
        return sps.block_diag(blocks, format='csr')


def array_mbk_linops(array, refn, f):

    MBK, MBK_inv = array_mbk_spmatrix(array, refn, f, inv=True)

    amesh = mesh.Mesh.from_abstract(array, refn=refn)
    ob = amesh.on_boundary
    nnodes = len(amesh.vertices)

    def mv(x):
        x[ob] = 0
        p = MBK.dot(x)
        p[ob] = 0
        return p
    linop = sps.linalg.LinearOperator(
        (nnodes, nnodes), dtype=np.complex128, matvec=mv)

    def inv_mv(x):
        x[ob] = 0
        p = MBK_inv.dot(x)
        p[ob] = 0
        return p
    linop_inv = sps.linalg.LinearOperator(
        (nnodes, nnodes), dtype=np.complex128, matvec=inv_mv)

    return linop, linop_inv


@util.memoize
def mem_patch_fcol_vector(mem, refn):
    '''
    '''
    f = mem_patch_f_matrix(mem, refn)
    avg = mem_patch_avg_matrix(mem, refn)

    K = mem_k_matrix(mem, refn)
    Kinv = inv_block(K)
    u = Kinv.dot(-np.sum(f, axis=1)).squeeze()
    u = u / np.max(np.abs(u))

    g = mem.gap

    fcol = []
    ups = []

    for i, pat in enumerate(mem.patches):

        # f_pat = f[:,i]
        # u = Kinv.dot(-f_pat).squeeze()
        # u = u / np.max(np.abs(u))
        avg_pat = avg[:, i]
        scale = -g / u[avg_pat > 0].min()
        # scale = -g / u.dot(avg_pat)
        up = scale * u
        up[up < -g] = -g

        p = (K.dot(up)).dot(avg_pat) / pat.area
        fcol.append(-p)
        ups.append(up)
        # fcol.append(-g / u[f_pat > 0].min())

    return np.array(fcol), np.array(ups)


if __name__ == '__main__':
    pass
