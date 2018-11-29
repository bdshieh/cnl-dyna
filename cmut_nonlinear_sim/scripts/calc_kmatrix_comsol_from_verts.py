'''
'''
import numpy as np
import argparse

# from cmut_nonlinear_sim import abstract
from cmut_nonlinear_sim import comsol


if __name__ == '__main__':

    # define arguments
    # verts = np.load('square.npy')
    lx = 40e-6
    ly = 40e-6
    lz = 2e-6
    rho = 2200
    ymod = 70e9
    pratio = 0.17
    fine = 2
    dx = lx / 10
    # r_start = 2
    # r_stop = 10
    
    comsol.connect_comsol()
    # K1 = comsol.square_membrane(verts, lx, ly, lz, rho, ymod, pratio, fine, dx)
    K = comsol.square_membrane_from_mesh(lx, ly, lz, rho, ymod, pratio, fine, dx, refn=4)

    # np.savez('kmatrix_test.npz', K1=K1, K2=K2)
    np.savez('kmatrix_test.npz', K=K)

    # refn = range(r_start, r_stop)
    # Ks = []
    # for r in refn:
        # K = comsol.square_membrane_from_mesh(lx, ly, lz, rho, ymod, pratio, fine, r)
        # Ks.append(K)

    # np.savez(file, K=Ks, refn=refn)