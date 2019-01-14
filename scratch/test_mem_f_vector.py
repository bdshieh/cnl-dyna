
import numpy as np
from matplotlib import pyplot as plt

from cnld import fem, mesh


def load_func(x, y):
    if x >= 40e-6 / 6:
        if x <= 40e-6 / 2:
            if y >= 40e-6 / 6:
                if y <= 40e-6 / 2:
                    return 1
    return 0
    # return 1

sqmesh = mesh.square(40e-6, 40e-6, refn=15)

f = fem.mem_f_vector_arb_load(sqmesh, load_func)

fi = mesh.interpolator(sqmesh, f, function='linear')
gridx, gridy = np.mgrid[-20e-6:20e-6:101j, -20e-6:20e-6:101j]

plt.imshow(fi(gridx, gridy))
plt.show()
