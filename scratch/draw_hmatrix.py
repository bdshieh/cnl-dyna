import numpy as np
from matplotlib import pyplot as plt

from cnld import fem, bem
from cnld.arrays import matrix_array




f = 1e6
c = 1500
k = 2 * np.pi *f / c
refn = 9

hmargs = {
    'aprx': 'paca',
    'basis': 'linear',
    'admis': '2',
    'eta': 0.8,
    'm': 4,
    'clf': 16,
    'eps_aca': 1e-2,
    'q_reg': 2,
    'q_sing': 4,
    'strict': True,
}

array = matrix_array(nelem=[1,2], shape='square')

Zhm = bem.array_z_matrix(array, refn, k, format='HFormat', **hmargs)
Zhm.draw()

