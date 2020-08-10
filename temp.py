#%%
import numpy as np
from matplotlib import pyplot as plt
from cnld.api import geometry

geom1 = geometry.circle_cmut_1mhz_geometry()
geom2 = geometry.square_cmut_1mhz_geometry()
geom = geom1 + geom2

print(geom)

#%%
from cnld.api import layout

lay = layout.matrix_layout(2, 2, 100e-6, 100e-6)
layout.generate_control_domains(lay, geom2)
# %%
import numpy as np
from cnld import datatypes

g = datatypes.GeometryData(id=0,
                           thickness=1e-6,
                           shape='square',
                           lengthx=35e-6,
                           lengthy=35e-6,
                           prat=np.arange(5),
                           eps_r=datatypes.GeometryData())
print(g)
# %%
