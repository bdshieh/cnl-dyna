

from cmut_nonlinear_sim.mesh import *
from cmut_nonlinear_sim.zmatrix import *
import numpy as np
# import scipy as sp
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

    ma = fast_matrix_array(5, 5, 60e-6, 60e-6, refn=5, lengthx=40e-6, lengthy=40e-6)
    # ma = matrix_array(5, 5, 60e-6, 60e-6, refn=5, shape='square', lengthx=40e-6, lengthy=40e-6)
    # ma.draw()

    # ma = matrix_array(4, 4, 60e-6, 60e-6, refn=3, shape='circle', radius=20e-6)
    # ma.draw()

    # ma = s
    k = 2 * np.pi * 10e6 / 1500.

    Z_hm = HierarchicalMatrix(ma, k, aprx='paca', admis='max', eta=1.0, eps=1e-12, m=4, clf=16, 
        eps_aca=1e-2, rk=0, q_reg=2, q_sing=4, strict=False)
    # Z_full = FullMatrix(ma, k)

    # print('Z_full:', Z_full.size, 'MB')
    # print('Z_hm:', Z_hm.size, 'MB')
    
    # matrix-vector product
    x = np.ones(len(ma.vertices))
    # b_full = Z_full.dot(x)
    b_hm = Z_hm.dot(x)

    # # plt.plot(np.abs(b))
    # # plt.plot(np.abs(b_dense))
    # # plt.show()

    # # solve using full
    # b = np.ones(len(ma.vertices))
    # LU = lu(Z_full)
    # x_full = lusolve(LU, b)

    # # solve hm by LU
    # LU = lu(Z_hm, eps=1e-12)
    # x_hm = lusolve(LU, b)

    # # by forward method


    # plt.plot(np.abs(x_full))
    # plt.plot(np.abs(x_hm))
    # plt.show()

    # rmse = np.sqrt(np.mean(np.abs(x_lu - x) ** 2))
    
