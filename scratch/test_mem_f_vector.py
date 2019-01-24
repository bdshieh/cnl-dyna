
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import dblquad

from cnld import fem, mesh
from cnld.arrays import matrix



# def load_func(x, y):
#     if x >= 40e-6 / 6:
#         if x <= 40e-6 / 2:
#             if y >= 40e-6 / 6:
#                 if y <= 40e-6 / 2:
#                     return 1
#     return 0


# sqmesh = mesh.square(40e-6, 40e-6, refn=15)

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

        da, _ = dblquad(load_func_psi_eta, 0, 1, 0, lambda x: 1 - x, epsrel=1e-1, epsabs=1e-1)

        f[tri] += 1 / 6 * da

    return f


from cnld import util

def memoize(func):
    '''
    Simple memoizer to cache repeated function calls.
    '''
    def ishashable(obj):
        try:
            hash(obj)
        except TypeError:
            return False
        return True
    
    def make_hashable(obj):
        if not ishashable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tostring()
            return str(obj)
        if isinstance(obj, float):
            return round(obj, 18)
        return obj

    memo = {}
    def decorator(*args):
        key = tuple(make_hashable(a) for a in args)
        if key not in memo:
            memo[key] = func(*args)
        return memo[key]
    return decorator


@memoize
def square_patch_f_vector(nodes, triangles, mlx, mly, px, py, plx, ply):
    print('called')
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
            # print('quad')

    return f

from scipy import sparse as sps

def f_from_abstract(array, refn):
    '''
    Construct load vector based on patches of abstract array.
    '''
    blocks = []
    for elem in array.elements:
        for mem in elem.membranes:
            sqmesh = mesh.square(mem.length_x, mem.length_y, refn=refn)

            f = np.zeros((len(sqmesh.vertices), len(mem.patches)))
            for i, pat in enumerate(mem.patches):
                
                ff = square_patch_f_vector(sqmesh.vertices, sqmesh.triangles, mem.length_x, mem.length_y, 
                        pat.position[0] - mem.position[0], pat.position[1] - mem.position[1], 
                        pat.length_x, pat.length_y)

                f[:,i] = ff

            blocks.append(f)
    
    return sps.block_diag(blocks, format='csc')


# f = fem.mem_f_vector_arb_load(sqmesh, load_func)
array = matrix.main(matrix.Config(), None)

F = f_from_abstract(array, refn=7)
# f = fem.f_from_abstract(array, refn=7)

# mem = array.elements[0].membranes[0]
# pat = mem.patches[0]

# def load_func(x, y):
#     if x >= (pat.position[0] - pat.length_x / 2):
#         if x <= (pat.position[0] + pat.length_x / 2):
#             if y >= (pat.position[1] - pat.length_y / 2):
#                 if y <= (pat.position[1] + pat.length_y / 2):
#                     return 1
#     return 0
# mem_mesh = mesh.square(mem.length_x, mem.length_y, refn=7, center=mem.position)


# fi = mesh.interpolator(sqmesh, f, function='linear')
# gridx, gridy = np.mgrid[-20e-6:20e-6:101j, -20e-6:20e-6:101j]
# plt.imshow(fi(gridx, gridy))
# plt.show()
