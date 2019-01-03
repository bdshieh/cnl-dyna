## surface3d.h ##


from . _basic cimport *


cdef extern from 'surface3d.h' nogil:
    
    struct _surface3d:
        uint vertices
        uint edges
        uint triangles
        real (* x)[3]
        uint (* e)[2]
        uint (* t)[3]
        uint (* s)[3]
        real (* n)[3]
        real * g
        real hmin
        real hmax

    ctypedef _surface3d surface3d
    ctypedef surface3d * psurface3d
    ctypedef const surface3d * pcsurface3d

    psurface3d new_surface3d(uint vertices, uint edges, uint triangles)
    void del_surface3d(psurface3d gr)
    void prepare_surface3d(psurface3d gr)
    psurface3d merge_surface3d(pcsurface3d gr1, pcsurface3d gr2)
    psurface3d refine_red_surface3d(psurface3d gr)
    void translate_surface3d(psurface3d gr, real * t)