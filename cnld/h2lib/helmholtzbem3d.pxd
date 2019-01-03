## helmholtzbem3d_cy.pxd ##


from . cimport _helmholtzbem3d
from . basic cimport *
from . surface3d cimport *
from . bem3d cimport *


cpdef new_slp_helmholtz_bem3d(field k, Surface3d surf, uint q_regular, uint q_singular,
    basisfunctionbem3d row_basis, basisfunctionbem3d col_basis)

cpdef new_dlp_helmholtz_bem3d(field k, Surface3d surf, uint q_regular, uint q_singular, 
    basisfunctionbem3d row_basis, basisfunctionbem3d col_basis, field alpha)


