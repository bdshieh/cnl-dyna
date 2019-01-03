## macrosurface3d.h ##

from . _basic cimport *
from . _surface3d cimport *

cdef extern from 'macrosurface3d.h' nogil:

    struct _macrosurface3d:
        uint vertices
        uint edges
        uint triangles
        real (* x)[3]
        uint (* e)[2]
        uint (* t)[3]
        uint (* s)[3]
        void (* phi) (uint i, real xr1, real xr2, void * phidata, real xt[3])
        void * phidata

    ctypedef _macrosurface3d macrosurface3d
    ctypedef macrosurface3d * pmacrosurface3d
    ctypedef const macrosurface3d * pcmacrosurface3d

    pmacrosurface3d new_macrosurface3d(uint vertices, uint edges, uint triangles)
    void del_macrosurface3d(pmacrosurface3d mg)
    psurface3d build_from_macrosurface3d_surface3d(pcmacrosurface3d mg, uint split) 
    pmacrosurface3d new_sphere_macrosurface3d()