## surface3d_cy.pxd ##


from . cimport _surface3d
from . basic cimport *


ctypedef _surface3d.psurface3d psurface3d
ctypedef _surface3d.pcsurface3d pcsurface3d

cdef class Surface3d:
    cdef psurface3d ptr
    cdef bint owner
    cdef real [:,:] _x
    cdef uint [:,:] _e
    cdef uint [:,:] _t
    cdef uint [:,:] _s
    cdef real [:,:] _n
    cdef real [:] _g
    cdef _setup(self, psurface3d ptr, bint owner)
    @staticmethod
    cdef wrap(psurface3d ptr, bint owner=*)

cpdef prepare_surface3d(Surface3d gr)
cpdef Surface3d merge_surface3d(Surface3d gr1, Surface3d gr2)
cpdef Surface3d refine_red_surface3d(Surface3d gr)
cpdef translate_surface3d(Surface3d gr, real[:] t)