    
import numpy as np
import numpy.linalg
import scipy.sparse as sps
import scipy as sp
import scipy.linalg 
from matplotlib import pyplot as plt
from tqdm import tqdm    

from cnld import util, bem, fem, mesh, abstract
from cnld.arrays import matrix
# from cnld.compressed_formats2 import MbkSparseMatrix, MbkFullMatrix


refn = 5
c = 1500.
freqs = np.arange(1e6, 50e6 + 1e6, 1e6)
f = 1e6

matkwargs = {}
matkwargs['density'] = [2040.,]
matkwargs['thickness'] = [2e-6,]
matkwargs['y_modulus'] = [110e9,]
matkwargs['p_ratio'] = [0.22,]
matkwargs['nelem'] = [1, 2]

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
hmkwargs['aprx'] = 'paca'
hmkwargs['rk'] = 0


array = matrix.main(matrix.Config(**matkwargs), None)
array_mesh = mesh.Mesh.from_abstract(array, refn)
ob = array_mesh.on_boundary

F = np.array(fem.f_from_abstract(array, refn).todense())


# x = np.zeros((nnodes, len(freqs)), dtype=np.complex128)
# for i, f in enumerate(tqdm(freqs)):

omg = 2 * np.pi * f
k = omg / c

Gfe, _ = fem.mbk_from_abstract(array, f, refn)
Gfe = np.array(Gfe.todense())
Z = bem.z_from_abstract(array, k, refn, format='FullFormat', **hmkwargs).data
Gbe = -omg**2 * 1000 * 2 * Z
G = Gfe + Gbe

lu, piv = sp.linalg.lu_factor(G)
# b = F[:,4]
b = np.ones(len(array_mesh.vertices))
b[ob] = 0
x = sp.linalg.lu_solve((lu, piv), b)
# x = np.linalg.solve(G, b)
x[ob] = 0
# x[:,i] = _x


fi = mesh.interpolator(array_mesh, x, function='cubic')

for elem in array.elements:
    for mem in elem.membranes:
        cx, cy, cz = mem.position
        lx = mem.length_x
        ly = mem.length_y
        gridx, gridy = np.mgrid[(cx - lx / 2):(cx + lx / 2):21j, (cy - ly / 2):(cy + ly / 2):21j]
        xi = fi(gridx, gridy)

        fig, ax = plt.subplots(figsize=(7,7))
        im = ax.imshow(np.abs(xi), cmap='RdBu_r')
        fig.colorbar(im)
        fig.show()

# fig, ax = plt.subplots(figsize=(7,7))
# ax.plot(gridx.ravel(), gridy.ravel(), '.')
# ax.set_aspect('equal')

fig.show()


# # fig, ax = plt.subplots(figsize=(7,7))
# # ax.plot(freqs, np.max(np.abs(x), axis=0))

# fig, ax = plt.subplots(figsize=(7,7))
# ax.plot(freqs, np.max(np.abs(x), axis=0))

# plt.show()
