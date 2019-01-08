    
import numpy as np
import numpy.linalg
import scipy.sparse as sps
from matplotlib import pyplot as plt
from tqdm import tqdm   
from scipy.interpolate import Rbf   

from cnld import util, bem, fem, mesh
from cnld.compressed_formats2 import MbkSparseMatrix


refn = 8
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
hmkwargs['m'] = 4
hmkwargs['q_reg'] = 2
hmkwargs['q_sing'] = 4
hmkwargs['admis'] = '2'
hmkwargs['eta'] = 1.1
hmkwargs['eps'] = 1e-12
hmkwargs['eps_aca'] = 1e-2
hmkwargs['strict'] = False
hmkwargs['clf'] = 16
hmkwargs['rk'] = 0
freqs = np.arange(500e3, 50e6 + 500e3, 500e3)

sqmesh = mesh.square(40e-6, 40e-6, refn=refn)
ob = sqmesh.on_boundary

M = fem.mem_m_matrix(sqmesh, rho, h)
K = fem.mem_k_matrix(sqmesh, E, h, eta)
F = fem.mem_f_vector(sqmesh, 1)
# F[ob] = 0

# from scipy.sparse.linalg import gmres, cg, bicg, LinearOperator

x = np.zeros((len(sqmesh.vertices), len(freqs)), dtype=np.complex128)
for i, f in enumerate(tqdm(freqs)):

    omg = 2 * np.pi * f
    k = omg / c

    # MBK = fem.mbk_from_mesh(sqmesh, f, rho, h, E, eta, amode, bmode, za, zb)

    # B = fem.mem_b_matrix_eig(sqmesh, M, K, amode, bmode, za, zb)
    # _MBK = -omg**2 * M + 1j * omg * B + K
    _MBK = -omg**2 * M + K
    _MBK[np.ix_(ob, ob)] = 0
    MBK = MbkSparseMatrix(_MBK)

    Z = bem.z_from_mesh(sqmesh, k, format='FullFormat', **hmkwargs)

    G = -(1000 * 2 * omg**2 * Z) + MBK
    G_LU = G.lu()
    # _x = np.linalg.solve(G[np.ix_(~ob, ~ob)], F[~ob])
    _x = G_LU.lusolve(F)

    # Z_LU = Z.lu()
    # def matvec(x):
    #     p1 = -(1000 * 2 * omg**2 * Z) * x
    #     p1[ob] = 0
    #     p2 = MBK * x
    #     p2[ob] = 0
    #     return p1 + p2
    # linop = LinearOperator(MBK.shape, matvec)
    # luop = LinearOperator(MBK.shape, G_LU._matvec)
    # _x, _  = gmres(linop, F, x0=np.ones(MBK.shape[0]), tol=1e-12, maxiter=20, M=luop)
    # del G, G_LU, Z
    
    x[~ob,i] = _x[~ob]

gridx, gridy = np.mgrid[-20e-6:20e-6:101j, -20e-6:20e-6:101j]

fi = Rbf(sqmesh.vertices[:,0], sqmesh.vertices[:,1], x[:, 10], function='cubic', smooth=0)
xi = fi(gridx, gridy)
fig, ax = plt.subplots(figsize=(7,7))
im = ax.imshow(np.abs(xi), cmap='RdBu_r')
fig.colorbar(im)
# ax.set_title(f'{f[4] / 1e6:0.2f} MHz')

fig, ax = plt.subplots(figsize=(7,7))
ax.plot(freqs, np.max(np.abs(x), axis=0))


plt.show()

# create MBK matrix in SparseFormat
# MBK = bem.mbk_from_abstract(array, f, refn, format='SparseFormat')
# MBK = fem.mbk_from_abstract(array, f, refn, format='SparseFormat')

# create Z matrix in HFormat

# hmargs = { k:getattr(cfg, k) for k in hmkwrds }
# Z = bem.z_from_abstract(array, k, refn, **hmargs)