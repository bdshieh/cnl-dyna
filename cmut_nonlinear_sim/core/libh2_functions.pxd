
from . libh2 cimport *
from . libh2_classes cimport *

cpdef build_from_macrosurface_surface(Macrosurface ms, uint refn)
cpdef new_slp_bem(field k, Surface surf, uint q_reg, uint q_sing, row_basis, col_basis)