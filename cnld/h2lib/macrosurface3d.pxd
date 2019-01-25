## macrosurface3d_cy.pxd ##


from . cimport _macrosurface3d
from . basic cimport *
from . surface3d cimport *


ctypedef _macrosurface3d.pmacrosurface3d pmacrosurface3d
ctypedef _macrosurface3d.pcmacrosurface3d pcmacrosurface3d

cdef class Macrosurface3d:
    cdef pmacrosurface3d ptr
    cdef bint owner
    cdef real [:,:] _x
    cdef uint [:,:] _e
    cdef uint [:,:] _t
    cdef uint [:,:] _s
    cdef _setup(self, pmacrosurface3d ptr, bint owner)
    @staticmethod
    cdef wrap(pmacrosurface3d ptr, bint owner=*)

cpdef build_from_macrosurface3d_surface3d(Macrosurface3d ms, uint refn)
cpdef Macrosurface3d new_sphere_macrosurface3d()