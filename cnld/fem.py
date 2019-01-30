'''
'''
import numpy as np
from numpy.linalg import inv, eig
from scipy import sparse as sps, linalg
from scipy.integrate import dblquad
from scipy.constants import epsilon_0

from cnld import util
from cnld.compressed_formats2 import ZHMatrix, ZFullMatrix, MbkSparseMatrix, MbkFullMatrix
from . mesh import square, Mesh


@util.memoize
def mem_k_matrix(mesh, E, h, eta):
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
        return np.array([[-nx, 0],[0, -ny],[-ny, -nx]])

    # get mesh information
    nodes = mesh.vertices
    triangles = mesh.triangles
    triangle_edges = mesh.triangle_edges
    triangle_areas = mesh.g / 2
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
    mesh.triangle_neighbors = triangle_neighbors

    # construct constitutive matrix for material
    h = h[0] # no support for composite membranes yet
    E = E[0]
    eta = eta[0]

    D = np.zeros((3,3))
    D[0,0] = 1
    D[0,1] = eta
    D[1,0] = eta
    D[1,1] = 1
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
        gradop = np.linalg.inv(J.T).dot([[-1, 1, 0],[-1, 0, 1]])
        # gradop = np.linalg.inv(J.T).dot([[1, 0, -1],[0, 1, -1]])

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
        for j, n in enumerate(neighbors):
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
        Bp = Bp / ap

        # construct local K matrix for control element
        Kp = (Bp.T).dot(D).dot(Bp) * ap

        # add matrix values to global K matrix
        K[np.ix_(Kidx, Kidx)] += Kp[np.ix_(Kpidx, Kpidx)]
    
    ob = mesh.on_boundary
    K[ob,:] = 0
    K[:, ob] = 0
    K[ob, ob] = 1

    return K


@util.memoize
def mem_m_matrix(mesh, rho, h, mu=0.5):
    '''
    Mass matrix based on average of lumped and consistent mass matrix (lumped-consistent).
    '''
    DLM = mem_dlm_matrix(mesh, rho, h)
    CMM = mem_cm_matrix(mesh, rho, h)

    return mu * DLM  + (1 - mu) * CMM


@util.memoize
def mem_cm_matrix(mesh, rho, h):
    '''
    Mass matrix based on kinetic energy and linear shape functions (consistent).
    '''
    # get mesh information
    nodes = mesh.vertices
    triangles = mesh.triangles

    mass = sum([x * y for x, y in zip(rho, h)])

    # construct M matrix by adding contribution from each element
    M = np.zeros((len(nodes), len(nodes)))
    for tt in range(len(triangles)):
        tri = triangles[tt,:]
        xi, yi = nodes[tri[0],:2]
        xj, yj = nodes[tri[1],:2]
        xk, yk = nodes[tri[2],:2]

        da = ((xj - xi) * (yk - yi) - (xk - xi) * (yj - yi))
        Mt = np.array([[1, 1 / 2, 1 / 2], [1 / 2, 1, 1 / 2], [1 / 2, 1 / 2, 1]]) / 12
        M[np.ix_(tri, tri)] += Mt * mass * da

    ob = mesh.on_boundary
    M[ob,:] = 0
    M[:, ob] = 0
    M[ob, ob] = 1

    return M


@util.memoize
def mem_dlm_matrix(mesh, rho, h):
    '''
    Mass matrix based on equal distribution of element mass to nodes (diagonally-lumped).
    '''
    # get mesh information
    nodes = mesh.vertices
    triangles = mesh.triangles
    triangle_areas = mesh.g / 2

    mass = sum([x * y for x, y in zip(rho, h)])

    # construct M matrix by adding contribution from each element
    M = np.zeros((len(nodes), len(nodes)))
    for tt in range(len(triangles)):
        tri = triangles[tt,:]
        ap = triangle_areas[tt]
        M[tri, tri] += 1 / 3 * mass * ap

    ob = mesh.on_boundary
    M[ob,:] = 0
    M[:, ob] = 0
    M[ob, ob] = 1

    return M


@util.memoize
def mem_b_matrix(M, K, fa, fb, za, zb):
    '''
    Damping matrix based on Rayleigh damping for damping ratios at two frequencies.
    '''
    omga = 2 * np.pi * fa
    omgb = 2 * np.pi * fb
    
    # solve for alpha and beta
    A = 1 / 2 * np.array([[1 / omga, omga],[1 / omgb, omgb]])
    alpha, beta = inv(A).dot([za, zb])
    
    return alpha * M + beta * K


@util.memoize
def mem_b_matrix_eig(mesh, M, K, amode, bmode, za, zb):
    '''
    Damping matrix based on Rayleigh damping for damping ratios at two modal frequencies.
    '''
    ob = mesh.on_boundary

    # determine eigenfrequencies of membrane
    w, v = eig(inv(M).dot(K)[np.ix_(~ob, ~ob)])
    omg = np.sort(np.sqrt(np.abs(w)))
    omga = omg[amode]
    omgb = omg[bmode]

    # solve for alpha and beta
    A = 1 / 2 * np.array([[1 / omga, omga],[1 / omgb, omgb]])
    alpha, beta = inv(A).dot([za, zb])
    
    return alpha * M + beta * K
    

@util.memoize    
def mem_f_vector(mesh, p):
    '''
    Pressure load vector based on equal distribution of pressure to element nodes.
    '''
    nodes = mesh.vertices
    triangles = mesh.triangles
    triangle_areas = mesh.g / 2
    # ob = mesh.on_boundary

    f = np.zeros(len(nodes))
    for tt in range(len(triangles)):
        tri = triangles[tt,:]
        ap = triangle_areas[tt]
        f[tri] += 1 / 3 * p * ap

    ob = mesh.on_boundary
    f[ob] = 0

    return f


def mem_f_vector_arb_load(mesh, load_func):
    '''
    Pressure load vector based on an arbitrary load.
    '''
    nodes = mesh.vertices
    triangles = mesh.triangles

    f = np.zeros(len(nodes))
    for tt in range(len(triangles)):
        tri = triangles[tt,:]
        xi, yi = nodes[tri[0],:2]
        xj, yj = nodes[tri[1],:2]
        xk, yk = nodes[tri[2],:2]

        def load_func_psi_eta(psi, eta):
            x = (xj - xi) * psi + (xk - xi) * eta + xi
            y = (yj - yi) * psi + (yk - yi) * eta + yi
            return load_func(x, y)

        da, _ = dblquad(load_func_psi_eta, 0, 1, 0, lambda x: 1 - x, epsabs=1e-1, epsrel=1e-1)
        f[tri] += 1 / 6 * da

    ob = mesh.on_boundary
    f[ob] = 0

    return f


@util.memoize
def square_patch_f_vector(nodes, triangles, on_boundary, mlx, mly, px, py, plx, ply):
    '''
    Load vector for a square patch.
    '''
    def load_func(x, y):
        if x >= (px - plx / 2):
            if x <= (px + plx / 2):
                if y >= (py - ply / 2):
                    if y <= (py + ply / 2):
                        return 1
        return 0
    
    f = np.zeros(len(nodes))
    for tt in range(len(triangles)):
        tri = triangles[tt,:]
        xi, yi = nodes[tri[0],:2]
        xj, yj = nodes[tri[1],:2]
        xk, yk = nodes[tri[2],:2]

        # check if triangle vertices are inside or outside load
        loadi = load_func(xi, yi)
        loadj = load_func(xj, yj)
        loadk = load_func(xk, yk)
        # if load covers entire triangle
        if all([loadi, loadj, loadk]):
            da = ((xj - xi) * (yk - yi) - (xk - xi) * (yj - yi))
            f[tri] += 1 / 6 * da
        # if load does not cover any part of triangle
        elif not any([loadi, loadj, loadk]):
            continue
        # if load partially covers triangle
        else:
            def load_func_psi_eta(psi, eta):
                x = (xj - xi) * psi + (xk - xi) * eta + xi
                y = (yj - yi) * psi + (yk - yi) * eta + yi
                return load_func(x, y)

            frac, _ = dblquad(load_func_psi_eta, 0, 1, 0, lambda x: 1 - x, epsrel=1e-1, epsabs=1e-1)
            da = ((xj - xi) * (yk - yi) - (xk - xi) * (yj - yi))
            f[tri] += 1 / 6 * frac * da

    f[on_boundary] = 0

    return f


def f_from_abstract(array, refn):
    '''
    Construct load vector based on patches of abstract array.
    '''
    blocks = []
    for elem in array.elements:
        for mem in elem.membranes:
            sqmesh = square(mem.length_x, mem.length_y, refn=refn)
            ob = sqmesh.on_boundary

            f = np.zeros((len(sqmesh.vertices), len(mem.patches)))
            for i, pat in enumerate(mem.patches):
                f[:,i]  = square_patch_f_vector(sqmesh.vertices, sqmesh.triangles, sqmesh.on_boundary,
                    mem.length_x, mem.length_y, pat.position[0] - mem.position[0], 
                    pat.position[1] - mem.position[1], pat.length_x, pat.length_y)
                
                f[ob,i] = 0
            blocks.append(f)
    
    return sps.block_diag(blocks, format='csc')


@util.memoize
def inv_block(a):
    return np.linalg.inv(a)


def mbk_from_abstract(array, f, refn):
    '''
    '''
    omg = 2 * np.pi * f
    blocks = []
    blocks_inv = []
    for elem in array.elements:
        for mem in elem.membranes:
            
            mesh = square(mem.length_x, mem.length_y, refn=refn)
            M = mem_m_matrix(mesh, mem.density, mem.thickness)
            K = mem_k_matrix(mesh, mem.y_modulus, mem.thickness, mem.p_ratio)
            B = mem_b_matrix_eig(mesh, M, K, mem.damping_mode_a, mem.damping_mode_b, 
                mem.damping_ratio_a, mem.damping_ratio_b)

            # block = -(omg**2) * M + 1j * omg * B + K
            block = -(omg**2) * M - 1j * omg * B + K
            # block = -(omg**2) * M + K
            block_inv = inv_block(block)
            blocks.append(block)
            blocks_inv.append(block_inv)
    
    return sps.block_diag(blocks, format='csr'), sps.block_diag(blocks_inv, format='csr')


# def mbk_from_mesh(mesh, f, rho, h, E, eta, amode, bmode, za, zb, format='SparseFormat',):
#     '''
#     '''
#     omg = 2 * np.pi * f

#     M = mem_m_matrix(mesh, rho, h)
#     K = mem_k_matrix(mesh, E, h, eta)
#     B = mem_b_matrix_eig(mesh, M, K, amode, bmode, za, zb)

#     MBK = -omg**2 + M + 1j * omg * B + K
#     if format.lower() in ['sparse', 'sparseformat']:
#         return MbkSparseMatrix(sps.csr_matrix(MBK))
#     else:
#         return MbkFullMatrix(MBK)


def mbk_linear_operators(array, f, refn):

    MBK, MBK_inv = mbk_from_abstract(array, f, refn)

    mesh = Mesh.from_abstract(array, refn=refn)
    ob = mesh.on_boundary
    nnodes = len(mesh.vertices)

    def mv(x):
        x[ob] = 0
        p = MBK.dot(x)
        p[ob] = 0
        return p
    linop = sps.linalg.LinearOperator((nnodes, nnodes), dtype=np.complex128, matvec=mv)

    def inv_mv(x):
        x[ob] = 0
        p = MBK_inv.dot(x)
        p[ob] = 0
        return p
    linop_inv = sps.linalg.LinearOperator((nnodes, nnodes), dtype=np.complex128, matvec=inv_mv)

    return linop, linop_inv


if __name__ == '__main__':
    pass