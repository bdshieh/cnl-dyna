
import numpy as np
import numpy.linalg
import scipy.sparse as sps
from matplotlib import pyplot as plt
from tqdm import tqdm
from scipy.signal import convolve2d

from cnld import util, fem, mesh


if __name__ == '__main__':

    E = 110e9
    h = 2e-6
    eta = 0.22
    refn = 7
    sqmesh = mesh.square(40e-6, 40e-6, refn=refn)
    ob = sqmesh.on_boundary
    K = fem.mem_k_matrix(sqmesh, E, h, eta)

    rho = 2040.
    CMM = fem.mem_m_matrix(sqmesh, rho, h)
    LMM = fem.mem_m_matrix2(sqmesh, rho, h)
    M = (CMM + LMM) / 2
    # idx = 3
    # with np.load('kmat1.npz') as npf:
    #     K = npf['Ks'][idx]
    #     ob = npf['on_bounds'][idx]
    #     refn = npf['refns'][idx]

    # B = 5e-10 * M + 5e-10 * K
    # B = fem.mem_b_matrix_eig(sqmesh, M, K, 0, 4, 0.004, 0.008)

    # E = 110e9
    # h = 2e-6
    # eta = 0.22
    # lx, ly = 40e-6, 40e-6
    # shape = 9, 9
    # nnodes = shape[0] * shape[1]
    # dx, dy = lx / (shape[0] - 1), ly / (shape[1] - 1)
    # K = k_matrix_fd2(E, h, eta, dx, dy, nnodes, shape)

    # rho = 2040
    # h = 2e-6
    # M = (sps.eye(K.shape[0]) * rho * h).todense()
    # M[ob] = 1e-9

    freqs = np.arange(500e3, 100e6 + 500e3, 500e3)
    F = fem.mem_f_vector(sqmesh, 1)
    # p = np.diag(fem.mem_m_matrix(sqmesh, 1, 1))
    # p = np.ones(K.shape[0])
    # p[ob] = 0
    # x = []
    from cnld.compressed_formats2 import MbkFullMatrix

    x = np.zeros((len(sqmesh.vertices), len(freqs)), dtype=np.complex128)
    for i, f in enumerate(tqdm(freqs)):
        # G = -(2 * np.pi * f)**2 * M + K + 1j * (2 * np.pi * f) * B
        G = -(2 * np.pi * f)**2 * M + K
        # G = MbkFullMatrix(G[np.ix_(~ob, ~ob)])
        # _x = G.lu().lusolve(F[~ob])
        # G[np.ix_(~ob, ~ob)] = 0
        _x = np.linalg.solve(G[np.ix_(~ob, ~ob)], F[~ob])
        x[~ob,i] = _x
        # _x = np.linalg.solve(G, F)
        # _x[ob] = 0
        # x[:,i] = _x
        # G = -(2 * np.pi * f)**2 * M[np.ix_(~ob, ~ob)] + K[np.ix_(~ob, ~ob)]
        # x.append(np.linalg.solve(G, p[~ob]))
    # x = np.array(x).T

    # from scipy.interpolate import Rbf
    # from scipy.signal import argrelextrema
    # args = argrelextrema(np.max(np.abs(x), axis=0), np.greater)[0]
    # fidx, fidx2 = args[:2]
    # # fidx = args[0]
    # # fidx = np.argmin(np.abs(freqs - 1e6))
    # # fidx = np.argmax(np.max(np.abs(x), axis=0))
    # # fidx2 = np.argmax(np.max(np.abs(x), axis=0)[(fidx + 1):]) + fidx + 1
    # gridx, gridy = np.mgrid[-20e-6:20e-6:101j, -20e-6:20e-6:101j]

    # # xi = griddata(sqmesh.vertices[:,:2], x[:,fidx], (gridx, gridy), method='nearest')
    # fi = Rbf(sqmesh.vertices[:,0], sqmesh.vertices[:,1], x[:, fidx], function='cubic', smooth=0)
    # xi = fi(gridx, gridy)
    # fig, ax = plt.subplots(figsize=(7,7))
    # im = ax.imshow(np.abs(xi), cmap='RdBu_r')
    # fig.colorbar(im)
    # ax.set_title(f'{freqs[fidx] / 1e6} MHz')

    # # xi2 = griddata(sqmesh.vertices[:,:2], x[:,fidx2], (gridx, gridy), method='nearest')
    # fi = Rbf(sqmesh.vertices[:,0], sqmesh.vertices[:,1], x[:, fidx2], function='cubic', smooth=0)
    # xi2 = fi(gridx, gridy)
    # fig, ax = plt.subplots(figsize=(7,7))
    # im = ax.imshow(np.abs(xi2), cmap='RdBu_r', aspect='equal')
    # fig.colorbar(im)
    # ax.set_title(f'{freqs[fidx2] / 1e6} MHz')
    # plt.show()

    # from mpl_toolkits.mplot3d import Axes3D
    # fig = plt.figure(figsize=(7,7))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(gridx, gridy, np.abs(xi), vmin=0, vmax=np.max(np.abs(xi)), cmap='RdBu_r')
    # # plt.show()

    # plt.figure(figsize=(7,7))
    # plt.plot(np.abs(xi)[xi.shape[0] // 2, :], '.-')
    # plt.plot(np.abs(xi)[:, xi.shape[1] // 2], '.-')
    # plt.show()

    # xo = x.reshape(())
    plt.figure(figsize=(7,7))
    plt.plot(freqs, np.abs(x).max(axis=0))
    plt.show()

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