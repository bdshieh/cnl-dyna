    
import numpy as np
import numpy.linalg
import scipy.sparse as sps
from matplotlib import pyplot as plt
from tqdm import tqdm   
from scipy.interpolate import Rbf   

from cnld import util, bem, fem, mesh
from cnld.compressed_formats2 import MbkSparseMatrix


refn = 4
c = 1500.
rho = 2040.
E = 110e9
h = 2.0e-6
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
freqs = np.arange(100e3, 50e6 + 100e3, 100e3)

sqmesh = mesh.square(40e-6, 40e-6, refn=refn)
ob = sqmesh.on_boundary

F = fem.mem_f_vector(sqmesh, 1)
F[ob] = 0

x = np.zeros((len(sqmesh.vertices), len(freqs)), dtype=np.complex128)
for i, f in enumerate(tqdm(freqs)):

    omg = 2 * np.pi * f
    k = omg / c

    # MBK = fem.mbk_from_mesh(sqmesh, f, rho, h, E, eta, amode, bmode, za, zb)
    M = fem.mem_m_matrix2(sqmesh, rho, h)
    K = fem.mem_k_matrix(sqmesh, E, h, eta)
    B = fem.mem_b_matrix_eig(sqmesh, M, K, amode, bmode, za, zb)
    _MBK = -omg**2 * M + 1j * omg * B + K
    _MBK[np.ix_(ob, ob)] = 0
    MBK = MbkSparseMatrix(_MBK)

    Z = bem.z_from_mesh(sqmesh, k, **hmkwargs)

    G = -(1000 * omg**2 * Z) + MBK
    G_LU = G.lu()
    # _x = np.linalg.solve(G[np.ix_(~ob, ~ob)], F[~ob])
    _x = G_LU.lusolve(F)

    # Z_LU = Z.lu()
    # def matvec(x):

        # _x = MBK[np.ix_(~ob, ~ob)].dot(x[~ob]) + (1j * omg * Z * x)[~ob]
        # _x

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