

from cmut_nonlinear_sim.mesh import *
from cmut_nonlinear_sim.zmatrix import *
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
import sys


if __name__ == '__main__':

    # if len(sys.argv[1:]):
        # refn = int(sys.argv[1])
    # else:
        # refn = 2

    # s = square(40e-6, 40e-6, refn=16)
    # s.draw()

    # c = circle(20e-6, refn=16)
    # c.draw()

    ma = matrix_array(5, 5, 60e-6, 60e-6, refn=5, shape='square', lengthx=40e-6, lengthy=40e-6)
    # ma.draw()

    # ma = matrix_array(4, 4, 60e-6, 60e-6, refn=3, shape='circle', radius=20e-6)
    # ma.draw()

    # ma = s
    k = 2 * np.pi * 1e6 / 1500.

    Z = HierarchicalMatrix(ma, k, aprx='paca', admis='max', eta=0.9, eps=1e-12, m=4, clf=64, 
        eps_aca=1e-19, rk=0, q_reg=2, q_sing=4)
    Z_dense = DenseMatrix(ma, k)

    print('Z_dense:', Z_dense.size, 'MB')
    print('Z_hmatrix:', Z.size, 'MB')
    
    x = np.ones(len(ma.vertices))
    b_dense = Z_dense.dot(x)
    b = Z.dot(x)

    # b = np.ones(len(ma.vertices))

    # Z_chol = chol(Z, eps=1e-6)
    # x_chol = cholsolve(Z_chol, b)

    # Z_lu = lu(Z_dense)
    # x_lu = lusolve(Z_lu, b)

    # rmse = np.sqrt(np.mean(np.abs(x_lu - x) ** 2))
    
    plt.plot(np.abs(b))
    plt.plot(np.abs(b_dense))
    plt.show()