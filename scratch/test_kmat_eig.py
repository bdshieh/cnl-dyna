
import numpy as np
import numpy.linalg
import scipy.sparse as sps
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm
from scipy.signal import convolve2d
from scipy.linalg import eig, inv
from scipy.interpolate import Rbf

from cnld import util, fem, mesh


if __name__ == '__main__':

    E = 110e9
    h = 2e-6
    eta = 0.22
    refn = 6
    sqmesh = mesh.square(40e-6, 40e-6, refn=refn)
    ob = sqmesh.on_boundary
    K = fem.mem_k_matrix(sqmesh, E, h, eta)

    rho = 2040.
    # M = fem.mem_m_matrix(sqmesh, rho, h)
    CMM = fem.mem_m_matrix(sqmesh, rho, h)
    LMM = fem.mem_m_matrix2(sqmesh, rho, h)
    M = (CMM + LMM) / 2


    w, v = eig(inv(M).dot(K)[np.ix_(~ob, ~ob)])
    sortix = np.argsort(np.abs(np.real(w)))
    f = np.sqrt(np.abs(w[sortix])) / (2 * np.pi)
    v = v[:, sortix]

    x = np.zeros((len(sqmesh.vertices), len(v)), dtype=np.complex128)
    x[~ob, :] = v

    
    # from scipy.signal import argrelextrema
    # args = argrelextrema(np.max(np.abs(x), axis=0), np.greater)[0]
    # fidx, fidx2 = args[:2]
  
    gridx, gridy = np.mgrid[-20e-6:20e-6:101j, -20e-6:20e-6:101j]

    for i in range(10):

        fi = Rbf(sqmesh.vertices[:,0], sqmesh.vertices[:,1], x[:, i], function='cubic', smooth=0)
        xi = fi(gridx, gridy)
        vmax = max(abs(xi.min()), abs(xi.max()))
        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(np.real(xi), vmin=-vmax, vmax=vmax, cmap='RdBu_r', interpolation='spline16')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)
        ax.set_title(f'Mode {i} at {f[i] / 1e6:0.2f} MHz')
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        plt.tight_layout()



    # fi = Rbf(sqmesh.vertices[:,0], sqmesh.vertices[:,1], x[:, fidx2], function='cubic', smooth=0)
    # xi2 = fi(gridx, gridy)
    # fig, ax = plt.subplots(figsize=(7,7))
    # im = ax.imshow(np.abs(xi2), cmap='RdBu_r', aspect='equal')
    # fig.colorbar(im)
    # ax.set_title(f'{freqs[fidx2] / 1e6} MHz')
    # # plt.show()

    # from mpl_toolkits.mplot3d import Axes3D
    # fig = plt.figure(figsize=(7,7))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(gridx, gridy, np.abs(xi), vmin=0, vmax=np.max(np.abs(xi)), cmap='RdBu_r')
    # # plt.show()

    # plt.figure(figsize=(7,7))
    # plt.plot(np.abs(xi)[xi.shape[0] // 2, :], '.-')
    # plt.plot(np.abs(xi)[:, xi.shape[1] // 2], '.-')
    # # plt.show()

    # # xo = x.reshape(())
    # plt.figure(figsize=(7,7))
    # plt.plot(freqs, np.abs(x).mean(axis=0))
    # plt.show()

    # import time
    # plt.figure()
    # for tri in sqmesh.triangles:
    #     i, j, k = tri
    #     ni = sqmesh.vertices[i,:]
    #     nj = sqmesh.vertices[j,:]
    #     nk = sqmesh.vertices[k,:]
    #     plt.plot(ni[0], ni[1], 'r.')
    #     plt.plot(nj[0], nj[1], 'g.')
    #     plt.plot(nk[0], nk[1], 'b.')
    #     plt.show()
    plt.show()