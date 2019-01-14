'''
'''
import numpy as np
from numpy.linalg import inv, eig
from scipy import sparse as sps, linalg
from scipy.integrate import dblquad

from cnld import util
from cnld.compressed_formats2 import ZHMatrix, ZFullMatrix, MbkSparseMatrix, MbkFullMatrix
from . mesh import square


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
    
    return K


@util.memoize
def mem_m_matrix(mesh, rho, h):
    '''
    Mass matrix based on kinetic energy and linear shape functions (consistent).
    '''
    # get mesh information
    nodes = mesh.vertices
    triangles = mesh.triangles

    # construct M matrix by adding contribution from each element
    M = np.zeros((len(nodes), len(nodes)))
    for tt in range(len(triangles)):
        tri = triangles[tt,:]
        xi, yi = nodes[tri[0],:2]
        xj, yj = nodes[tri[1],:2]
        xk, yk = nodes[tri[2],:2]

        da = ((xj - xi) * (yk - yi) - (xk - xi) * (yj - yi))
        Mt = np.array([[1, 1 / 2, 1 / 2], [1 / 2, 1, 1 / 2], [1 / 2, 1 / 2, 1]]) / 12
        M[np.ix_(tri, tri)] += Mt * rho * h * da

    return M


@util.memoize
def mem_m_matrix2(mesh, rho, h):
    '''
    Mass matrix based on equal distribution of element mass to nodes (diagonally-lumped).
    '''
    # get mesh information
    nodes = mesh.vertices
    triangles = mesh.triangles
    triangle_areas = mesh.g / 2

    # construct M matrix by adding contribution from each element
    M = np.zeros((len(nodes), len(nodes)))
    for tt in range(len(triangles)):
        tri = triangles[tt,:]
        ap = triangle_areas[tt]
        M[tri, tri] += 1 / 3 * rho * h * ap

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

    return f


# @util.memoize    
# def mem_f_vector2(mesh, p):
#     '''
#     Pressure load vector based on equal distribution of pressure to element nodes.
#     '''
#     nodes = mesh.vertices
#     triangles = mesh.triangles
#     # triangle_areas = mesh.g / 2
#     # ob = mesh.on_boundary

#     f = np.zeros(len(nodes))
#     for tt in range(len(triangles)):
#         tri = triangles[tt,:]
#         # ap = triangle_areas[tt]
#         xi, yi = nodes[tri[0],:2]
#         xj, yj = nodes[tri[1],:2]
#         xk, yk = nodes[tri[2],:2]

#         da = ((xj - xi) * (yk - yi) - (xk - xi) * (yj - yi))

#         f[tri] += 1 / 6 * p * da

#     return f

def mem_f_vector_arb_load(mesh, load_func):
    '''
    Pressure load vector based on equal distribution of pressure to element nodes.
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

        da, _ = dblquad(load_func_psi_eta, 0, 1, 0, lambda x: 1 - x)

        f[tri] += 1 / 6 * da

    return f


def f_from_abstract(array, refn):
    '''
    '''
    blocks = []
    for elem in array.elements:
        for mem in elem.membranes:
            mesh = square(mem.length_x, mem.length_y, refn=refn)

            f = sps.csc_matrix((len(mesh.vertices), len(mem.patches)))
            for i, pat in enumerate(mem.patches):
                def load_func(x, y):
                    if x >= pat.position[0] - pat.length_x / 2:
                        if x <= pat.position[0] + pat.length_x / 2:
                            if y >= pat.position[1] - pat.length_y / 2:
                                if y <= pat.position[1] + pat.length_y / 2:
                                    return 1
                    return 0
                
                f[:,i] = mem_f_vector_arb_load(mesh, load_func)
            
            blocks.append(f)
    
    return sps.block_diag(blocks)



def mbk_from_abstract(array, f, refn, format='SparseFormat'):
    '''
    '''
    omg = 2 * np.pi * f
    blocks = []
    for elem in array.elements:
        for mem in elem.membranes:
            
            mesh = square(mem.length_x, mem.length_y, refn=refn)
            M = mem_m_matrix(mesh, mem.density, mem.thickness)
            K = mem_k_matrix(mesh, mem.y_modulus, mem.thickness, mem.p_ratio)
            B = mem_b_matrix_eig(mesh, M, K, mem.damping_mode_a, mem.damping_mode_a, 
                mem.damping_ratio_a, mem.damping_ratio_b)

            block = -(omg**2) * M + 1j * omg * B + K
            blocks.append(block)
    
    if format.lower() in ['sparse', 'sparseformat']:
        return MbkSparseMatrix(sps.csr_matrix(sps.block_diag(blocks)))
    else:
        return MbkFullMatrix(sps.block_diag(blocks).todense())


def mbk_from_mesh(mesh, f, rho, h, E, eta, amode, bmode, za, zb, format='SparseFormat',):
    '''
    '''
    omg = 2 * np.pi * f

    M = mem_m_matrix(mesh, rho, h)
    K = mem_k_matrix(mesh, E, h, eta)
    B = mem_b_matrix_eig(mesh, M, K, amode, bmode, za, zb)

    MBK = -omg**2 + M + 1j * omg * B + K
    if format.lower() in ['sparse', 'sparseformat']:
        return MbkSparseMatrix(sps.csr_matrix(MBK))
    else:
        return MbkFullMatrix(MBK)


if __name__ == '__main__':

    from cnld.mesh import square

    E = 110e9
    h = 2e-6
    eta = 0.22

    mesh = square(40e-6, 40e-6, refn=4)
    K = mem_k_matrix(mesh, E, h, eta)