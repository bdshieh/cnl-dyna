    
import numpy as np
import numpy.linalg
import scipy.sparse as sps
from matplotlib import pyplot as plt
from tqdm import tqdm   
from scipy.interpolate import Rbf  
import scipy as sp
import scipy.linalg 

from cnld import util, bem, fem, mesh
from cnld.compressed_formats2 import MbkSparseMatrix, MbkFullMatrix


refn = 7
c = 1500.
rho = 2040.
E = 110e9
h = 2e-6
eta = 0.22
amode = 0
bmode = 4
za = 0.004
zb = 0.006
hmkwargs = {}
hmkwargs['basis'] = 'linear'
hmkwargs['q_reg'] = 2
hmkwargs['q_sing'] = 4
hmkwargs['admis'] = '2'
hmkwargs['eta'] = 1.1
hmkwargs['eps'] = 1e-12
hmkwargs['eps_aca'] = 1e-2
hmkwargs['strict'] = True
hmkwargs['clf'] = 16
hmkwargs['aprx'] = 'aca'
hmkwargs['rk'] = 0
freqs = np.arange(500e3, 50e6 + 500e3, 500e3)

sqmesh = mesh.square(40e-6, 40e-6, refn=refn)
ob = sqmesh.on_boundary
nnodes = len(sqmesh.vertices)

x = np.zeros((nnodes, len(freqs)), dtype=np.complex128)
# xmax = []
for i, f in enumerate(tqdm(freqs)):

    omg = 2 * np.pi * f
    k = omg / c
    wl = c / f

    CMM = fem.mem_m_matrix(sqmesh, rho, h)
    LMM = fem.mem_m_matrix2(sqmesh, rho, h)
    M = (CMM + LMM) / 2

    K = fem.mem_k_matrix(sqmesh, E, h, eta)

    MBK = -omg**2 * M + K
    MBK_inv = np.linalg.inv(MBK)

    def _Gfe(x):
        x[ob] = 0
        p = MBK.dot(x)
        p[ob] = 0
        return p
    Gfe = sps.linalg.LinearOperator((nnodes, nnodes), dtype=np.complex128, matvec=_Gfe)

    def _Gfe_inv(x):
        x[ob] = 0
        p = MBK_inv.dot(x)
        p[ob] = 0
        return p
    Gfe_inv = sps.linalg.LinearOperator((nnodes, nnodes), dtype=np.complex128, matvec=_Gfe_inv)
    
    # Z = bem.z_from_mesh(sqmesh, k, format='FullFormat', **hmkwargs)
    Z = bem.z_from_mesh(sqmesh, k, format='HFormat', **hmkwargs)
    # break
    Z_LU = Z.lu()
    # from scipy.linalg import lu_factor, lu_solve
    # LU, PIV = lu_factor(Z._mat.a)
    
    def _Gbe(x):
        x[ob] = 0
        p = Z * x
        p[ob] = 0
        return -omg**2 * 1000 * 2 * p
    Gbe = sps.linalg.LinearOperator((nnodes, nnodes), dtype=np.complex128, matvec=_Gbe)

    def _Gbe_inv(x):
        x[ob] = 0
        p = Z_LU._triangularsolve(x)
        # p = lu_solve((LU, PIV), x)
        p[ob] = 0
        return -omg**2 * 1000 * 2 * p
    Gbe_inv = sps.linalg.LinearOperator((nnodes, nnodes), dtype=np.complex128, matvec=_Gbe_inv)

    F = fem.mem_f_vector(sqmesh, 1)
    F[ob] = 0

    G = Gfe + Gbe
    P = Gbe_inv * Gfe_inv

    _x, _ = sps.linalg.lgmres(G, F, tol=1e-12, maxiter=40, M=P)
    x[:,i] = _x
    # break

# gridx, gridy = np.mgrid[-20e-6:20e-6:101j, -20e-6:20e-6:101j]

# fi = Rbf(sqmesh.vertices[:,0], sqmesh.vertices[:,1], x[:, 10], function='cubic', smooth=0)
# xi = fi(gridx, gridy)
# fig, ax = plt.subplots(figsize=(7,7))
# im = ax.imshow(np.abs(xi), cmap='RdBu_r')
# fig.colorbar(im)
# # ax.set_title(f'{f[4] / 1e6:0.2f} MHz')

# fig, ax = plt.subplots(figsize=(7,7))
# ax.plot(freqs, np.max(np.abs(x), axis=0))

# fig, ax = plt.subplots(figsize=(7,7))
# ax.plot(freqs, np.max(np.abs(x), axis=0))

# plt.show()

# create MBK matrix in SparseFormat
# MBK = bem.mbk_from_abstract(array, f, refn, format='SparseFormat')
# MBK = fem.mbk_from_abstract(array, f, refn, format='SparseFormat')

# create Z matrix in HFormat

# hmargs = { k:getattr(cfg, k) for k in hmkwrds }
# Z = bem.z_from_abstract(array, k, refn, **hmargs)