import numpy as np
from matplotlib import pyplot as plt

from cnld import fem, bem, mesh
from cnld.arrays import matrix_array




f = 1e6
c = 1500
k = 2 * np.pi *f / c
refn = 8

hmargs = {
    'aprx': 'paca',
    'basis': 'linear',
    'admis': '2',
    'eta': 1.2,
    'm': 4,
    'clf': 16,
    'eps_aca': 1e-2,
    'q_reg': 2,
    'q_sing': 4,
    'strict': False,
}

array = matrix_array(nelem=[20,20], shape='circle')
amesh = mesh.Mesh.from_abstract(array, refn)

print(mesh.check_surface3d(amesh._surface))
print(mesh.isclosed_surface3d(amesh._surface))
print(mesh.isoriented_surface3d(amesh._surface))

Zhm = bem.array_z_matrix(array, refn, k, format='HFormat', **hmargs)
# Zhm.draw()

