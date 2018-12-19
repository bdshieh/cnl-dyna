'''
'''
import numpy as np


def mem_k_matrix(mesh, E, h, eta):

    def none_case1(f):
        def decorator(p):
            if p is None:
                return 0
            else:
                return f(p)
        return decorator

    def none_case2(f):
        def decorator(p, o):
            if o is None:
                return 0
            else:
                return f(p, o)
        return decorator

    @none_case1
    def bibar(p):
        tri = triangles[p,:]
        ap = triangle_areas[p]
        yj = nodes[tri[1],1]
        yk = nodes[tri[2],1]
        return (yj - yk) / (2 * ap)

    @none_case1 
    def bjbar(p):
        tri = triangles[p,:]
        ap = triangle_areas[p]
        yi = nodes[tri[0],1]
        yk = nodes[tri[2],1]
        return (yk - yi) / (2 * ap)

    @none_case1
    def bkbar(p):
        tri = triangles[p,:]
        ap = triangle_areas[p]
        yi = nodes[tri[0],1]
        yj = nodes[tri[1],1]
        return (yi - yj) / (2 * ap)

    @none_case2
    def blbar(p, b):
        tri = triangles[p,:]
        ap = triangle_areas[b]
        yi = nodes[tri[1],1]
        yj = nodes[tri[1],1]
        return (yj - yi) / (2 * ap)

    @none_case2
    def bmbar(p, c):
        tri = triangles[p,:]
        ap = triangle_areas[c]
        yj = nodes[tri[1],1]
        yk = nodes[tri[2],1]
        return (yk - yj) / (2 * ap)

    @none_case2
    def bnbar(p, d):
        tri = triangles[p,:]
        ap = triangle_areas[d]
        yi = nodes[tri[1],1]
        yk = nodes[tri[2],1]
        return (yi - yk) / (2 * ap)

    @none_case1
    def cibar(p):
        tri = triangles[p,:]
        ap = triangle_areas[p]
        xj = nodes[tri[1],0]
        xk = nodes[tri[2],0]
        return (xk - xj) / (2 * ap)

    @none_case1
    def cjbar(p):
        tri = triangles[p,:]
        ap = triangle_areas[p]
        xi = nodes[tri[0],0]
        xk = nodes[tri[2],0]
        return (xi - xk) / (2 * ap)

    @none_case1
    def ckbar(p):
        tri = triangles[p,:]
        ap = triangle_areas[p]
        xi = nodes[tri[0],0]
        xj = nodes[tri[1],0]
        return (xj - xi) / (2 * ap)
    
    @none_case2
    def clbar(p, b):
        tri = triangles[p,:]
        ap = triangle_areas[b]
        xi = nodes[tri[0],0]
        xj = nodes[tri[1],0]
        return (xi - xj) / (2 * ap)

    @none_case2
    def cmbar(p, c):
        tri = triangles[p,:]
        ap = triangle_areas[c]
        xj = nodes[tri[1],0]
        xk = nodes[tri[2],0]
        return (xj - xk) / (2 * ap)

    @none_case2
    def cnbar(p, d):
        tri = triangles[p,:]
        ap = triangle_areas[d]
        xi = nodes[tri[0],0]
        xk = nodes[tri[2],0]
        return (xk - xi) / (2 * ap)

    nodes = mesh.vertices
    edges = mesh.edges
    triangles = mesh.triangles
    triangle_edges = mesh.triangle_edges
    triangle_areas = mesh.g / 2
    on_bound = mesh.on_boundary

    triangle_neighbors = np.ones((len(triangles), 3)) * np.nan
    for tt in range(len(triangles)):
        for i, te in enumerate(triangle_edges[tt,:]):
            mask = np.any(triangle_edges == te, axis=1)
            args = np.nonzero(mask)[0]
            if len(args) > 1:
                triangle_neighbors[tt,i] = args[args != tt][0]
    D = np.zeros((3,3))
    D[0,0] = 1
    D[0,1] = eta
    D[1,0] = eta
    D[2,2] = (1 - eta) / 2
    D = D * E * h**3 / (12 * (1 - eta**2))

    K = np.zeros((len(nodes), len(nodes)))
    for p in range(len(triangles)):
        tri = triangles[p,:]
        ap = triangle_areas[p]
        xi, yi, _ = nodes[tri[0],:]
        xj, yj, _ = nodes[tri[1],:]
        xk, yk, _ = nodes[tri[2],:]
        neighbors = triangle_neighbors[p,:]
        c = int(neighbors[0]) if not np.isnan(neighbors[0]) else None
        d = int(neighbors[1]) if not np.isnan(neighbors[1]) else None
        b = int(neighbors[2]) if not np.isnan(neighbors[2]) else None

        bound_edge = np.zeros(3, dtype=np.bool)
        for i, ee in enumerate(triangle_edges[p,:]):
            bound_edge[i] = True if on_bound[edges[ee,0]] and on_bound[edges[ee,1]] else False
        jkbound, kibound, ijbound = bound_edge

        B = np.zeros((3, 6))
        B[0,0] = (yi - yj) * bibar(b) + (yk - yi) * bibar(d) if not jkbound else 0
        B[0,1] = (yi - yj) * bjbar(b) + (yj - yk) * bjbar(c) if not kibound else 0
        B[0,2] = (yj - yk) * bkbar(c) + (yk - yi) * bkbar(d) if not ijbound else 0
        B[0,3] = (yi - yj) * blbar(p, b)
        B[0,4] = (yj - yk) * bmbar(p, c)
        B[0,5] = (yk - yi) * bnbar(p, d)
        B[1,0] = -(xi - xj) * cibar(b) - (xk - xi) * cibar(d) if not jkbound else 0
        B[1,1] = -(xi - xj) * cjbar(b) - (xj - xk) * cjbar(c) if not kibound else 0
        B[1,2] = -(xj - xk) * ckbar(c) - (xk - xi) * ckbar(d) if not ijbound else 0
        B[1,3] = -(xi - xj) * clbar(p, b)
        B[1,4] = -(xj - xk) * cmbar(p, c)
        B[1,5] = -(xk - xi) * cnbar(p, d)
        B[2,0] = (yi - yj) * cibar(b) - (xi - xj) * bibar(b) + (yk - yi) * cibar(d) - (xk - xi) * bibar(d) if not jkbound else 0
        B[2,1] = (yi - yj) * cjbar(b) - (xj - xk) * bjbar(b) + (yj - yk) * cjbar(c) - (xj - xk) * bjbar(c) if not kibound else 0
        B[2,2] = (yj - yk) * ckbar(c) - (xj - xk) * bkbar(c) + (yk - yi) * ckbar(d) - (xk - xi) * bkbar(d) if not ijbound else 0
        B[2,3] = (yi - yj) * clbar(p, b) - (xi - xj) * blbar(p, b)
        B[2,4] = (yj - yk) * cmbar(p, c) - (xj - xk) * bmbar(p, c)
        B[2,5] = (yk - yi) * cnbar(p, d) - (xk - xi) * cnbar(p, d)
        B = B / ap

        Kp = (B.T).dot(D).dot(B) * ap

        # idx = np.nonzero(triangle_edges[trib,:] == triangle_edges[trip,2])[0]
        # yl = nodes[trib[idx],1]
        # trib = triangles[b,:]

        i, j, k = tri
        Kidx = [i, j, k]
        Kpidx = [0, 1, 2]

        if b is not None:
            trib = triangles[b,:]
            idx = np.nonzero(triangle_edges[b,:] == triangle_edges[p,2])[0]
            l = trib[idx]
            Kidx.append(l)
            Kpidx.append(3)
        
        if c is not None:
            tric = triangles[c,:]
            idx = np.nonzero(triangle_edges[c,:] == triangle_edges[p,0])[0]
            m = tric[idx]
            Kidx.append(m)
            Kpidx.append(4)

        if d is not None:
            trid = triangles[d,:]
            idx = np.nonzero(triangle_edges[d,:] == triangle_edges[p,1])[0]
            n = trid[idx]
            Kidx.append(n)
            Kpidx.append(5)

        K[np.ix_(Kidx, Kidx)] += Kp[np.ix_(Kpidx, Kpidx)]
    
    return K


def mem_k_matrix2(mesh, E, h, eta):
    '''
    '''
    def L(x1, y1, x2, y2):
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    def T_matrix(x1, y1, x2, y2):
        z = [0, 0, 1]
        r = [x2 - x1, y2 - y1]
        n = np.cross(z, r)
        n /= np.linalg.norm(n)
        nx, ny, _ = n
        return np.array([[-nx, 0],[0, -ny],[-ny, -nx]])

    nodes = mesh.vertices
    # edges = mesh.edges
    triangles = mesh.triangles
    triangle_edges = mesh.triangle_edges
    triangle_areas = mesh.g / 2
    # on_bound = mesh.on_boundary
    # nnodes = len(nodes)
    ntriangles = len(triangles)

    # determine list of neighbors for each triangle
    # None indicates neighbor doesn't exist for that edge (boundary edge)
    triangle_neighbors = []
    for tt in range(ntriangles):
        neighbors = []
        for te in triangle_edges[tt,:]:
            mask = np.any(triangle_edges == te, axis=1)
            args = np.nonzero(mask)[0]
            if len(args) > 1:
                neighbors.append(args[args != tt][0])
            else:
                neighbors.append(None)
        triangle_neighbors.append(neighbors)

    # construct constitutive matrix for material
    D = np.zeros((3,3))
    D[0,0] = 1
    D[0,1] = eta
    D[1,0] = eta
    D[2,2] = (1 - eta) / 2
    D = D * E * h**3 / (12 * (1 - eta**2))

    # calculate Jacobian and gradient operator for each triangle
    gradops = []
    for tt in range(ntriangles):
        tri = triangles[tt,:]
        xi, yi = nodes[tri[0],:2]
        xj, yj = nodes[tri[1],:2]
        xk, yk = nodes[tri[2],:2]

        J = np.array([[xj - xi, xk - xi], [yj - yi, yk - yi]])
        gradop = np.linalg.inv(J.T).dot([[1, 0, -1],[0, 1, -1]])

        gradops.append(gradop)

    # construct K matrix
    K = np.zeros((len(nodes), len(nodes)))
    for p in range(ntriangles):
        trip = triangles[p,:]
        ap = triangle_areas[p]

        xi, yi = nodes[trip[0],:2]
        xj, yj = nodes[trip[1],:2]
        xk, yk = nodes[trip[2],:2]

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
        for j in range(3):
            n = neighbors[j]
            if n is None:
                continue
            
            # determine index of the node in the neighbor opposite edge
            iin, jjn, kkn = triangles[n,:]
            uidx = [x for x in np.unique([ii, jj, kk, iin, jjn, kkn]) if x not in [ii, jj, kk]][0]
            # update indexes
            Kidx.append(uidx)
            Kpidx.append(3 + j)

            l = L(*edges[j])
            T = T_matrix(*edges[j])
            gradn = gradops[n]

            pterm = l / 2 * T.dot(gradp)
            Bp[:,:3] += pterm

            nterm = l / 2 * T.dot(gradn)
            idx = [Kpidx[Kidx.index(x)] for x in [iin, jjn, kkn]]
            Bp[:,idx] += nterm

        Bp /= ap

        # construct local K matrix for control element
        Kp = (Bp.T).dot(D).dot(Bp) * ap

        # add matrix values to global K matrix
        K[np.ix_(Kidx, Kidx)] += Kp[np.ix_(Kpidx, Kpidx)]
    
    return K


def mem_m_matrix(mesh, rho, h):

    nodes = mesh.vertices
    triangles = mesh.triangles
    triangle_areas = mesh.g / 2

    M = np.zeros((len(nodes), len(nodes)))
    for tt in range(len(triangles)):
        tri = triangles[tt,:]
        ap = triangle_areas[tt]
        M[tri, tri] += 1 / 3 * rho * h * ap

    return M


if __name__ == '__main__':

    from cnld.mesh import square

    E = 110e9
    h = 2e-6
    eta = 0.22

    mesh = square(40e-6, 40e-6, refn=4)
    K = mem_k_matrix(mesh, E, h, eta)