

import numpy as np
from matplotlib import pyplot as plt

from cnld import abstract, mesh


# m = mesh.geometry_square(40e-6, 40e-6, refn=2)
# m = mesh.matrix_array(2, 2, 60e-6, 60e-6, 40e-6, 40e-6, refn=2)

m = mesh.Mesh.from_abstract(abstract.load('matrix.json'), refn=7)
m.draw()