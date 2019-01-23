    
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
rho = [2040.,]
h = [2e-6,]
E = [110e9,]
eta = [0.22,]
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
freqs = np.arange(200e3, 50e6 + 200e3, 200e3)

sqmesh = mesh.square(40e-6, 40e-6, refn=refn)
ob = sqmesh.on_boundary
nnodes = len(sqmesh.vertices)

x = np.zeros((nnodes, len(freqs)), dtype=np.complex128)

for i, f in enumerate(tqdm(freqs)):

    omg = 2 * np.pi * f
    k = omg / c
    wl = c / f

    M = fem.mem_dlm_matrix(sqmesh, rho, h)
    K = fem.mem_k_matrix(sqmesh, E, h, eta)

    Gfe = -omg**2 * M + K
    Z = bem.z_from_mesh(sqmesh, k, format='FullFormat', **hmkwargs)
    Gbe = -omg ** 2 * 1000 * 2 * Z

    G = MbkSparseMatrix(Gfe) + Gbe
    Glu = G.lu()

    F = fem.mem_f_vector(sqmesh, 1)
    # F[ob] = 0

    _x = Glu.lusolve(F)
    _x[ob] = 0
    x[:,i] = _x

gridx, gridy = np.mgrid[-20e-6:20e-6:101j, -20e-6:20e-6:101j]

fi = Rbf(sqmesh.vertices[:,0], sqmesh.vertices[:,1], x[:, 10], function='cubic', smooth=0)
xi = fi(gridx, gridy)
fig, ax = plt.subplots(figsize=(7,7))
im = ax.imshow(np.abs(xi), cmap='RdBu_r')
fig.colorbar(im)
# ax.set_title(f'{f[4] / 1e6:0.2f} MHz')

# fig, ax = plt.subplots(figsize=(7,7))
# ax.plot(freqs, np.max(np.abs(x), axis=0))

fig, ax = plt.subplots(figsize=(7,7))
ax.plot(freqs, np.max(np.abs(x), axis=0))

plt.show()

np.savez(f'febe_refn_{refn}.npz', freqs=freqs, x=x, refn=refn)
# create MBK matrix in SparseFormat
# MBK = bem.mbk_from_abstract(array, f, refn, format='SparseFormat')
# MBK = fem.mbk_from_abstract(array, f, refn, format='SparseFormat')

# create Z matrix in HFormat

# hmargs = { k:getattr(cfg, k) for k in hmkwrds }
# Z = bem.z_from_abstract(array, k, refn, **hmargs)