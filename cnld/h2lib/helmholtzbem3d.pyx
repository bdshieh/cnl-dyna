## helmholtzbem3d_cy.pyx ##


from . cimport _helmholtzbem3d
from . basic cimport *
from . surface3d cimport *
from . bem3d cimport *


cpdef new_slp_helmholtz_bem3d(field k, Surface3d surf, uint q_regular, uint q_singular,
    basisfunctionbem3d row_basis, basisfunctionbem3d col_basis):

    cdef pbem3d bem = _helmholtzbem3d.new_slp_helmholtz_bem3d(k, <pcsurface3d> surf.ptr, q_regular, q_singular, 
        row_basis, col_basis)
    return Bem3d.wrap(bem, True)


cpdef new_dlp_helmholtz_bem3d(field k, Surface3d surf, uint q_regular, uint q_singular, 
    basisfunctionbem3d row_basis, basisfunctionbem3d col_basis, field alpha):

    cdef pbem3d bem = _helmholtzbem3d.new_dlp_helmholtz_bem3d(k, <pcsurface3d> surf.ptr, q_regular, q_singular, 
        row_basis, col_basis, alpha)
    return Bem3d.wrap(bem, True)


