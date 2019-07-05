

import numpy as np
import scipy as sp
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline

from cnld import abstract, arrays, compensation


refn = 7
E = 110e9
v = 0.22
Estar = E / (2 * (1 - v**2))
A = np.pi * (20e-6)**2

array = arrays.matrix_array(nelem=[1, 1], npatch=[2, 4], shape='circle')

cont_stiff = 2 * Estar / np.sqrt(np.pi * A) * 1
fcomp, meta = compensation.array_patch_fcomp_funcs(array, refn,
                                                   cont_stiff=cont_stiff)


x = np.linspace(-70e-9, 70e-9, 101)
v = 50
fig, ax = plt.subplots()
# ax.plot(x, fcomp[0](x, v), '.-')
ax.plot(x, fcomp[0](x, v), '--')
ax.plot(meta[0]['u'], meta[0]['f_es'] * v**2, '.-')
ax.plot(meta[0]['u'], meta[0]['f_cont'], '.-')
# ax.plot(x, fcomp[0][0](x) * v**2, '--')
# ax.plot(x, fcomp[0][1](x), '--')
# ax.plot(x, fcomp[0][0](x) * v**2 + fcomp[0][1](x), '--')
ax.plot(meta[0]['u'], meta[0]['f_es'] * v**2 + meta[0]['f_cont'], '.')
fig.show()
