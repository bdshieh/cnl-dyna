''''''
import numpy as np
import numpy.linalg
import numba


def k_mat_np(grid, geom, type='bpt'):
    '''
    Stiffness matrix.
    '''
    if type.lower() in ['bpt',]:
        return mem_k_matrix_bpt(mem, refn)
    else:
        return mem_k_matrix_hpb(mem, refn)


def k_bpt_mat_np(grid, geom):
    '''
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

    # get mesh information
    nodes = grid.vertices
    triangles = grid.triangles
    triangle_edges = grid.triangle_edges
    triangle_areas = grid.triangle_areas
    triangle_neighbors = grid.triangle_neighbors
    ntriangles = len(triangles)
    ob = grid.on_boundary

    # construct constitutive matrix for material
    h = geom.thickness
    E = geom.y_modulus
    eta = geom.p_ratio

    D = np.zeros((3, 3))
    D[0, 0] = 1
    D[0, 1] = eta
    D[1, 0] = eta
    D[1, 1] = 1
    D[2, 2] = (1 - eta) / 2
    D = D * E * h**3 / (12 * (1 - eta**2))

    # calculate Jacobian and gradient operator for each triangle
    gradops = [None] * ntriangles
    for tt in range(ntriangles):
        tri = triangles[tt, :]
        xi, yi = nodes[tri[0], :2]
        xj, yj = nodes[tri[1], :2]
        xk, yk = nodes[tri[2], :2]

        J = np.array([[xj - xi, xk - xi], [yj - yi, yk - yi]])
        gradop = np.linalg.inv(J.T).dot([[-1, 1, 0], [-1, 0, 1]])
        # gradop = np.linalg.inv(J.T).dot([[1, 0, -1],[0, 1, -1]])

        gradops[tt] = gradop

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
            uidx = [
                x for x in np.unique([ii, jj, kk, iin, jjn, kkn])
                if x not in [ii, jj, kk]
            ][0]
            # update indexes
            Kidx.append(uidx)
            Kpidx.append(3 + j)

            l = L(*edges[j])
            T = T_matrix(*edges[j])
            gradn = gradops[n]

            pterm = l / 2 * T @ gradp
            Bp[:, :3] += pterm

            nterm = l / 2 * T @ gradn
            idx = [Kpidx[Kidx.index(x)] for x in [iin, jjn, kkn]]
            Bp[:, idx] += nterm
        Bp = Bp / ap

        # construct local K matrix for control element
        Kp = Bp.T @ D @ Bp * ap

        # add matrix values to global K matrix
        K[np.ix_(Kidx, Kidx)] += Kp[np.ix_(Kpidx, Kpidx)]

    K[ob, :] = 0
    K[:, ob] = 0
    K[ob, ob] = 1

    return K


def k_hpb_mat_np(grid, geom):
    '''
    Stiffness matrix based on 3-dof hinged plate bending (HPB) elements.

    References
    ----------
    [1] R. Phaal and C. R. Calladine, Int. J. Numer. Meth. Engng.,
    vol. 35, no. 5, pp. 955–977, (1992).
    '''
    def norm(r1, r2):
        # calculates edge length
        return np.sqrt((r2[0] - r1[0])**2 + (r2[1] - r1[1])**2)

    # get mesh information
    nodes = grid.vertices
    triangles = grid.triangles
    triangle_edges = grid.triangle_edges
    triangle_areas = grid.triangle_areas
    ntriangles = len(triangles)
    ob = grid.on_boundary

    # determine list of neighbors and neighbor nodes for each triangle
    # None indicates neighbor doesn't exist for that edge (boundary edge)
    triangle_neighbors = []
    neighbors_node = []

    for tt in range(ntriangles):

        neighbors = []
        neighbor_node = []
        for te in triangle_edges[tt, :]:

            mask = np.any(triangle_edges == te, axis=1)
            args = np.nonzero(mask)[0]

            if len(args) > 1:

                n = args[args != tt][0]
                neighbors.append(n)

                # determine index of the node in the neighbor opposite edge
                iin, jjn, kkn = triangles[n, :]
                ii, jj, kk = triangles[tt, :]
                uidx = [
                    x for x in np.unique([ii, jj, kk, iin, jjn, kkn])
                    if x not in [ii, jj, kk]
                ][0]
                neighbor_node.append(uidx)

            else:
                neighbors.append(None)
                neighbor_node.append(None)

        triangle_neighbors.append(neighbors)
        neighbors_node.append(neighbor_node)

    grid.triangle_neighbors = triangle_neighbors
    grid.neighbors_node = neighbors_node

    # construct constitutive matrix for material
    h = geom.thickness
    E = geom.y_modulus
    eta = geom.p_ratio

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

        # assign primary triangle nodes based on order of edges
        x5, y5 = nodes[trip[0], :2]
        x6, y6 = nodes[trip[1], :2]
        x4, y4 = nodes[trip[2], :2]

        r4 = np.array([x4, y4])
        r5 = np.array([x5, y5])
        r6 = np.array([x6, y6])

        # assign neighboring triangle nodes and add fictitious nodes if necessary
        neighbors = neighbors_node[p]

        if neighbors[0] is None:
            x = r5
            xo = r6
            n = (r4 - r6) / norm(r4, r6)
            x1, y1 = -x + 2 * xo + 2 * (x - xo).dot(n) * n
        else:
            x1, y1 = nodes[neighbors[0], :2]

        if neighbors[1] is None:
            x = r6
            xo = r4
            n = (r5 - r4) / norm(r5, r4)
            x2, y2 = -x + 2 * xo + 2 * (x - xo).dot(n) * n
        else:
            x2, y2 = nodes[neighbors[1], :2]

        if neighbors[2] is None:
            x = r4
            xo = r5
            n = (r6 - r5) / norm(r6, r5)
            x3, y3 = -x + 2 * xo + 2 * (x - xo).dot(n) * n
        else:
            x3, y3 = nodes[neighbors[2], :2]

        r1 = np.array([x1, y1])
        r2 = np.array([x2, y2])
        r3 = np.array([x3, y3])

        # construct C matrix
        C = np.zeros((6, 3))
        C[0, :] = [x1**2 / 2, y1**2 / 2, x1 * y1]
        C[1, :] = [x2**2 / 2, y2**2 / 2, x2 * y2]
        C[2, :] = [x3**2 / 2, y3**2 / 2, x3 * y3]
        C[3, :] = [x4**2 / 2, y4**2 / 2, x4 * y4]
        C[4, :] = [x5**2 / 2, y5**2 / 2, x5 * y5]
        C[5, :] = [x6**2 / 2, y6**2 / 2, x6 * y6]

        # construct L matrix

        # calculate vars based on geometry of first sub-element
        L1 = norm(r6, r4)
        b1a1 = (r1 - r4).dot(r6 - r4) / L1
        b1a2 = (r1 - r6).dot(r4 - r6) / L1
        b1b1 = (r5 - r4).dot(r6 - r4) / L1
        b1b2 = (r5 - r6).dot(r4 - r6) / L1
        h1a = np.sqrt(norm(r1, r4)**2 - b1a1**2)
        h1b = np.sqrt(norm(r5, r6)**2 - b1b2**2)

        # calculate vars based on geometry of second sub-element
        L2 = norm(r5, r4)
        b2a1 = (r2 - r5).dot(r4 - r5) / L2
        b2a2 = (r2 - r4).dot(r5 - r4) / L2
        b2b1 = (r6 - r5).dot(r4 - r5) / L2
        b2b2 = (r6 - r4).dot(r5 - r4) / L2
        h2a = np.sqrt(norm(r2, r5)**2 - b2a1**2)
        h2b = np.sqrt(norm(r6, r4)**2 - b2b2**2)

        # calculate vars based on geometry of third sub-element
        L3 = norm(r5, r6)
        b3a1 = (r3 - r6).dot(r5 - r6) / L3
        b3a2 = (r3 - r5).dot(r6 - r5) / L3
        b3b1 = (r4 - r6).dot(r5 - r6) / L3
        b3b2 = (r4 - r5).dot(r6 - r5) / L3
        h3a = np.sqrt(norm(r3, r6)**2 - b3a1**2)
        h3b = np.sqrt(norm(r4, r5)**2 - b3b2**2)

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

        # calculate G matrix
        G = L @ C
        Ginv = np.linalg.inv(G)

        # create I_D matrix
        I_D = np.zeros((3, 3))
        I_D[0, 0] = 1
        I_D[1, 1] = 1
        I_D[2, 2] = 2

        # K_be = ap * (L.T).dot(Ginv.T).dot(I_D).dot(D).dot(Ginv).dot(L)
        K_be = ap * L.T @ Ginv.T @ I_D @ D @ Ginv @ L

        # begin putting together indexes needed later for matrix assignment
        K_idx = [trip[2], trip[0], trip[1]]
        K_be_idx = [3, 4, 5]

        # apply BCs
        if neighbors[0] is None:
            # fictitious node index = 0
            # mirrored node index = 4
            # boundary nodes index = 3, 5
            # non-boundary nodes index = 1, 2

            # modify row for mirrored node
            K_be[4,
                 4] = (K_be[0, 0] + K_be[0, 4] + K_be[4, 0] + K_be[4, 4])  #/ 2
            K_be[4, 1] = (K_be[4, 1] + K_be[0, 1])  #/ 2
            K_be[4, 2] = (K_be[4, 2] + K_be[0, 2])  #/ 2
            # K_be[4, 1] /= 2
            # K_be[1, 4] /= 2
            # K_be[4, 2] /= 2
            # K_be[2, 4] /= 2
            # K_be[4, 3] = (K_be[4, 3] + K_be[0, 3]) / 2
            # K_be[4, 5] = (K_be[4, 5] + K_be[0, 5]) / 2

            # modify row of first non-boundary node
            K_be[1, 4] = (K_be[1, 4] + K_be[1, 0])  #/ 2

            # modify row of second non-boundary node
            K_be[2, 4] = (K_be[2, 4] + K_be[2, 0])  #/ 2

            # modify row of first boundary node
            # K_be[3, 3] = K_be[3, 3] / 2
            # K_be[3, 4] = (K_be[3, 4] + K_be[3, 0]) / 2

            # modify row of second boundary node
            # K_be[5, 5] = K_be[5, 5] / 2
            # K_be[5, 4] = (K_be[5, 4] + K_be[5, 0]) / 2
        else:

            K_idx.append(neighbors[0])
            K_be_idx.append(0)

        if neighbors[1] is None:
            # fictitious node index = 1
            # mirrored node index = 5
            # boundary nodes index = 3, 4
            # non-boundary nodes index = 0, 2

            # modify row for mirrored node
            K_be[5,
                 5] = (K_be[1, 1] + K_be[1, 5] + K_be[5, 1] + K_be[5, 5])  #/ 2
            K_be[5, 0] = (K_be[5, 0] + K_be[1, 0])  #/ 2
            K_be[5, 2] = (K_be[5, 2] + K_be[1, 2])  #/ 2
            # K_be[5, 0] /= 2
            # K_be[0, 5] /= 2
            # K_be[5, 2] /= 2
            # K_be[2, 5] /= 2
            # K_be[5, 3] = (K_be[5, 3] + K_be[1, 3]) / 2
            # K_be[5, 4] = (K_be[5, 4] + K_be[1, 4]) / 2

            # modify row of first non-boundary node
            K_be[0, 5] = (K_be[0, 5] + K_be[0, 1])  #/ 2

            # modify row of second non-boundary node
            K_be[2, 5] = (K_be[2, 5] + K_be[2, 1])  #/ 2

            # modify row of first boundary node
            # K_be[3, 3] = K_be[3, 3] / 2
            # K_be[3, 5] = (K_be[3, 5] + K_be[3, 1]) / 2

            # modify row of second boundary node
            # K_be[4, 4] = K_be[4, 4] / 2
            # K_be[4, 5] = (K_be[4, 5] + K_be[4, 1]) / 2
        else:

            K_idx.append(neighbors[1])
            K_be_idx.append(1)

        if neighbors[2] is None:
            # fictitious node index = 2
            # mirrored node index = 3
            # boundary nodes index = 4, 5
            # non-boundary nodes index = 0, 1

            # modify row for mirrored node
            K_be[3,
                 3] = (K_be[2, 2] + K_be[2, 3] + K_be[3, 2] + K_be[3, 3])  #/ 2
            K_be[3, 0] = (K_be[3, 0] + K_be[2, 0])  #/ 2
            K_be[3, 1] = (K_be[3, 1] + K_be[2, 1])  #/ 2
            # K_be[3, 0] /= 2
            # K_be[0, 3] /= 2
            # K_be[3, 1] /= 2
            # K_be[1, 3] /= 2
            # K_be[3, 4] = (K_be[3, 4] + K_be[2, 4]) / 2
            # K_be[3, 5] = (K_be[3, 5] + K_be[2, 5]) / 2

            # modify row of first non-boundary node
            K_be[0, 3] = (K_be[0, 3] + K_be[0, 2])  #/ 2

            # modify row of second non-boundary node
            K_be[1, 3] = (K_be[1, 3] + K_be[1, 2])  #/ 2
        else:

            K_idx.append(neighbors[2])
            K_be_idx.append(2)

        # add matrix values to global K matrix
        K[np.ix_(K_idx, K_idx)] += K_be[np.ix_(K_be_idx, K_be_idx)]

    K[ob, :] = 0
    K[:, ob] = 0
    K[ob, ob] = 1

    return K


def k_hyb_mat_np(grid, geom):
    '''
    *Experimental* Stiffness matrix based on HPB for interior triangles and
    BPT for boundary triangles.
    '''
    def norm(r1, r2):
        # calculates edge length
        return np.sqrt((r2[0] - r1[0])**2 + (r2[1] - r1[1])**2)

    def LL(x1, y1, x2, y2):
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

    # get mesh information
    nodes = grid.vertices
    triangles = grid.triangles
    triangle_edges = grid.triangle_edges
    triangle_areas = grid.triangle_areas
    ntriangles = len(triangles)
    ob = grid.on_boundary

    # determine list of neighbors and neighbor nodes for each triangle
    # None indicates neighbor doesn't exist for that edge (boundary edge)
    triangle_neighbors = []
    neighbors_node = []

    for tt in range(ntriangles):

        neighbors = []
        neighbor_node = []
        for te in triangle_edges[tt, :]:

            mask = np.any(triangle_edges == te, axis=1)
            args = np.nonzero(mask)[0]

            if len(args) > 1:

                n = args[args != tt][0]
                neighbors.append(n)

                # determine index of the node in the neighbor opposite edge
                iin, jjn, kkn = triangles[n, :]
                ii, jj, kk = triangles[tt, :]
                uidx = [
                    x for x in np.unique([ii, jj, kk, iin, jjn, kkn])
                    if x not in [ii, jj, kk]
                ][0]
                neighbor_node.append(uidx)

            else:
                neighbors.append(None)
                neighbor_node.append(None)

        triangle_neighbors.append(neighbors)
        neighbors_node.append(neighbor_node)

    grid.triangle_neighbors = triangle_neighbors
    grid.neighbors_node = neighbors_node

    # construct constitutive matrix for material
    h = geom.thickness # no support for composite membranes yet
    E = geom.y_modulus
    eta = geom.p_ratio

    D = np.zeros((3, 3))
    D[0, 0] = 1
    D[0, 1] = eta
    D[1, 0] = eta
    D[1, 1] = 1
    D[2, 2] = (1 - eta)
    D = D * E * h**3 / (12 * (1 - eta**2))

    D2 = np.zeros((3, 3))
    D2[0, 0] = 1
    D2[0, 1] = eta
    D2[1, 0] = eta
    D2[1, 1] = 1
    D2[2, 2] = (1 - eta) / 2
    D2 = D2 * E * h**3 / (12 * (1 - eta**2))

    # calculate Jacobian and gradient operator for each triangle
    gradops = []
    for tt in range(ntriangles):
        tri = triangles[tt, :]
        xi, yi = nodes[tri[0], :2]
        xj, yj = nodes[tri[1], :2]
        xk, yk = nodes[tri[2], :2]

        J = np.array([[xj - xi, xk - xi], [yj - yi, yk - yi]])
        gradop = np.linalg.inv(J.T).dot([[-1, 1, 0], [-1, 0, 1]])

        gradops.append(gradop)

    # construct K matrix
    K = np.zeros((len(nodes), len(nodes)))

    for p in range(ntriangles):

        trip = triangles[p, :]
        ap = triangle_areas[p]

        # assign primary triangle nodes based on order of edges
        x5, y5 = nodes[trip[0], :2]
        x6, y6 = nodes[trip[1], :2]
        x4, y4 = nodes[trip[2], :2]

        r4 = np.array([x4, y4])
        r5 = np.array([x5, y5])
        r6 = np.array([x6, y6])

        # assign neighboring triangle nodes and add fictitious nodes if necessary
        neighbors = neighbors_node[p]

        if None in neighbors or ob[trip[0]] or ob[trip[1]] or ob[trip[2]]:
            # if True:

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
                uidx = [
                    x for x in np.unique([ii, jj, kk, iin, jjn, kkn])
                    if x not in [ii, jj, kk]
                ][0]
                # update indexes
                Kidx.append(uidx)
                Kpidx.append(3 + j)

                l = LL(*edges[j])
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

        else:
            x1, y1 = nodes[neighbors[0], :2]
            x2, y2 = nodes[neighbors[1], :2]
            x3, y3 = nodes[neighbors[2], :2]

            r1 = np.array([x1, y1])
            r2 = np.array([x2, y2])
            r3 = np.array([x3, y3])

            # construct C matrix
            C = np.zeros((6, 3))
            C[0, :] = [x1**2 / 2, y1**2 / 2, x1 * y1]
            C[1, :] = [x2**2 / 2, y2**2 / 2, x2 * y2]
            C[2, :] = [x3**2 / 2, y3**2 / 2, x3 * y3]
            C[3, :] = [x4**2 / 2, y4**2 / 2, x4 * y4]
            C[4, :] = [x5**2 / 2, y5**2 / 2, x5 * y5]
            C[5, :] = [x6**2 / 2, y6**2 / 2, x6 * y6]

            # construct L matrix

            # calculate vars based on geometry of first sub-element
            L1 = norm(r6, r4)
            b1a1 = (r1 - r4).dot(r6 - r4) / L1
            b1a2 = (r1 - r6).dot(r4 - r6) / L1
            b1b1 = (r5 - r4).dot(r6 - r4) / L1
            b1b2 = (r5 - r6).dot(r4 - r6) / L1
            h1a = np.sqrt(norm(r1, r4)**2 - b1a1**2)
            h1b = np.sqrt(norm(r5, r6)**2 - b1b2**2)

            # calculate vars based on geometry of second sub-element
            L2 = norm(r5, r4)
            b2a1 = (r2 - r5).dot(r4 - r5) / L2
            b2a2 = (r2 - r4).dot(r5 - r4) / L2
            b2b1 = (r6 - r5).dot(r4 - r5) / L2
            b2b2 = (r6 - r4).dot(r5 - r4) / L2
            h2a = np.sqrt(norm(r2, r5)**2 - b2a1**2)
            h2b = np.sqrt(norm(r6, r4)**2 - b2b2**2)

            # calculate vars based on geometry of third sub-element
            L3 = norm(r5, r6)
            b3a1 = (r3 - r6).dot(r5 - r6) / L3
            b3a2 = (r3 - r5).dot(r6 - r5) / L3
            b3b1 = (r4 - r6).dot(r5 - r6) / L3
            b3b2 = (r4 - r5).dot(r6 - r5) / L3
            h3a = np.sqrt(norm(r3, r6)**2 - b3a1**2)
            h3b = np.sqrt(norm(r4, r5)**2 - b3b2**2)

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

            # calculate G matrix
            G = L @ C
            Ginv = np.linalg.inv(G)

            # create I_D matrix
            I_D = np.zeros((3, 3))
            I_D[0, 0] = 1
            I_D[1, 1] = 1
            I_D[2, 2] = 2

            # K_be = ap * (L.T).dot(Ginv.T).dot(I_D).dot(D).dot(Ginv).dot(L)
            K_be = ap * L.T @ Ginv.T @ I_D @ D @ Ginv @ L

            # begin putting together indexes needed later for matrix assignment
            K_idx = [
                neighbors[0], neighbors[1], neighbors[2], trip[2], trip[0],
                trip[1]
            ]
            K_be_idx = [0, 1, 2, 3, 4, 5]

            # add matrix values to global K matrix
            K[np.ix_(K_idx, K_idx)] += K_be[np.ix_(K_be_idx, K_be_idx)]

    K[ob, :] = 0
    K[:, ob] = 0
    K[ob, ob] = 1

    return K

