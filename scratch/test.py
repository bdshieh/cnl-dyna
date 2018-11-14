

from cmut_nonlinear_sim.mesh import *
from cmut_nonlinear_sim.zmatrix import *
import numpy as np
from matplotlib import pyplot as plt
import sys


if __name__ == '__main__':

    # if len(sys.argv[1:]):
        # refn = int(sys.argv[1])
    # else:
        # refn = 2

    s = square(40e-6, 40e-6, refn=16)
    # s.draw()

    c = circle(20e-6, refn=16)
    # c.draw()

    # ma = matrix_array(3, 3, 60e-6, 60e-6, refn=3, shape='square', lengthx=40e-6, lengthy=40e-6)
    # ma.draw()

    ma = matrix_array(4, 4, 60e-6, 60e-6, refn=3, shape='circle', radius=20e-6)
    # ma.draw()

    # ma = s
    k = 1e6
    Z = HierarchicalMatrix(ma, k, clf=64, rk=32, eps=1e-9)
    # y = Z._matvec(np.ones(len(ma.vertices)))
    
    x = AVector(len(ma.vertices))
    y = AVector(len(ma.vertices))
    random_avector(x)
    clear_avector(y)
    
    # addeval_hmatrix_avector(1.0, Z._hmatrix, x, y)