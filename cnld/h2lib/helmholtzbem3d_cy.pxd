## helmholtzbem3d_cy.pxd ##


from . cimport helmholtzbem3d as _helmholtzbem3d
from . basic_cy cimport *
from . surface3d_cy cimport *
from . bem3d_cy cimport *


cpdef new_slp_helmholtz_bem3d(field k, Surface3d surf, uint q_regular, uint q_singular,
    basisfunctionbem3d row_basis, basisfunctionbem3d col_basis)

cpdef new_dlp_helmholtz_bem3d(field k, Surface3d surf, uint q_regular, uint q_singular, 
    basisfunctionbem3d row_basis, basisfunctionbem3d col_basis, field alpha)


