import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
import sys
from tqdm import tqdm
from timeit import default_timer as timer

from cmut_nonlinear_sim.mesh import *
from cmut_nonlinear_sim.zmatrix import *


def benchmark(Z):
    
    b = np.ones(Z.shape[0])
    
    # solve full
    start = timer()
    LU = lu(Z, eps=1e-9)
    time_lu = timer() - start
    
    start = timer()
    x = lusolve(LU, b)
    time_solve = timer() - start
    
    results = {}
    results['x'] = x
    results['size'] = Z.size
    results['time_assemble'] = Z.assemble_time
    results['time_lu'] = time_lu
    results['time_solve'] = time_solve
    
    del Z
    
    return results


def nrmse(x, xhat):
    return np.sqrt(np.mean(np.abs(x - xhat) ** 2)) / np.sqrt(np.sum(np.abs(xhat) ** 2))


def main(*args):

    opts = {}
    opts['aprx'] = 'paca'
    opts['basis'] = 'linear'
    opts['admis'] = 'max'
    opts['eta'] = 1.1
    opts['eps'] = 1e-12
    opts['m'] = 4
    opts['clf'] = 16
    opts['eps_aca'] = 1e-2
    opts['rk'] = 0
    opts['q_reg'] = 2
    opts['q_sing'] = 4
    opts['strict'] = False

    k = 2 * np.pi * 1e6 / 1500
    mesh = fast_matrix_array(5, 5, 60e-6, 60e-6, refn=3, lengthx=40e-6, lengthy=40e-6)

    hm = benchmark(HierarchicalMatrix(mesh, k, **opts))
    full = benchmark(FullMatrix(mesh, k))

    results = {}
    results['hm'] = hm
    results['full'] = full
    results['nrmse'] = nrmse(full['x'], hm['x'])
    results['vertices'] = len(mesh.vertices)
    results['edges'] = len(mesh.edges)
    results['triangles'] = len(mesh.triangles)
    
    print(results)


if __name__ == '__main__':
    
    import sys
    main(*sys.argv)