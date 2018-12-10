## macrosurface3d_cy.pxd ##


from . cimport macrosurface3d as _macrosurface3d
from . basic_cy cimport *
from . surface3d_cy cimport *


ctypedef _macrosurface3d.pmacrosurface3d pmacrosurface3d
ctypedef _macrosurface3d.pcmacrosurface3d pcmacrosurface3d

cdef class Macrosurface3d:
    cdef pmacrosurface3d ptr
    cdef bint owner
    cdef public real [:,:] x
    cdef public uint [:,:] e
    cdef public uint [:,:] t
    cdef public uint [:,:] s
    cdef _setup(self, pmacrosurface3d ptr, bint owner)
    @staticmethod
    cdef wrap(pmacrosurface3d ptr, bint owner=*)

cpdef build_from_macrosurface3d_surface3d(Macrosurface3d ms, uint refn)
cpdef Macrosurface3d new_sphere_macrosurface3d()