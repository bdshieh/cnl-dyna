## helmholtzbem3d.h ##

from . basic cimport *
from . surface3d cimport *
from . bem3d cimport *


cdef extern from 'helmholtzbem3d.h' nogil:

    pbem3d new_slp_helmholtz_bem3d(field k, pcsurface3d gr, uint q_regular, uint q_singular, 
        basisfunctionbem3d row_basis, basisfunctionbem3d col_basis)
    pbem3d new_dlp_helmholtz_bem3d(field k, pcsurface3d gr, uint q_regular, uint q_singular, 
        basisfunctionbem3d row_basis, basisfunctionbem3d col_basis, field alpha)