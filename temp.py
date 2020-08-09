#%%
import numpy as np
from matplotlib import pyplot as plt
from cnld.api import geometry

geom1 = geometry.circle_cmut_1mhz_geometry()
geom2 = geometry.square_cmut_1mhz_geometry()
geom = geom1 + geom2

#%%
from cnld.api import layout

lay = layout.matrix_layout(2, 2, 100e-6, 100e-6)

# %%
