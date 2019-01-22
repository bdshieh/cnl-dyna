## surface3d_cy.pyx ##


from . cimport _surface3d
from . basic cimport *


cdef class Surface3d():

    def __cinit__(self):
        self.ptr = NULL
        self.owner = False

    def __init__(self, uint vertices, uint edges, uint triangles):
        cdef psurface3d ptr = _surface3d.new_surface3d(vertices, edges, triangles)
        self._setup(ptr, True)

    def __dealloc__(self):
        if self.ptr is not NULL and self.owner is True:
            _surface3d.del_surface3d(self.ptr)

    cdef _setup(self, psurface3d ptr, bint owner):
        self.ptr = ptr
        self.owner = owner
        self.x = <real [:ptr.vertices,:3]> (<real *> ptr.x)
        self.e = <uint [:ptr.edges,:2]> (<uint *> ptr.e)
        self.t = <uint [:ptr.triangles,:3]> (<uint *> ptr.t)
        self.s = <uint [:ptr.triangles,:3]> (<uint *> ptr.s)
        self.n = <real [:ptr.triangles,:3]> (<real *> ptr.n)
        self.g = <real [:ptr.triangles]> (<real *> ptr.g)

    @property
    def vertices(self):
        return self.ptr.vertices

    @property
    def edges(self):
        return self.ptr.edges

    @property
    def triangles(self):
        return self.ptr.triangles

    # @property
    # def g(self):
    #     return self.ptr.g[0]

    @property
    def hmin(self):
        return self.ptr.hmin
    
    @property
    def hmax(self):
        return self.ptr.hmax
        
    @staticmethod
    cdef wrap(psurface3d ptr, bint owner=True):
        cdef Surface3d obj = Surface3d.__new__(Surface3d)
        obj._setup(ptr, owner)
        return obj

cpdef prepare_surface3d(Surface3d gr):
    _surface3d.prepare_surface3d(gr.ptr)

cpdef Surface3d merge_surface3d(Surface3d gr1, Surface3d gr2):
    cdef psurface3d surf = _surface3d.merge_surface3d(<pcsurface3d> gr1.ptr, <pcsurface3d> gr2.ptr)
    return Surface3d.wrap(surf, True)

cpdef Surface3d refine_red_surface3d(Surface3d gr):
    cdef psurface3d surf = _surface3d.refine_red_surface3d(gr.ptr)
    return Surface3d.wrap(surf, True)

cpdef translate_surface3d(Surface3d gr, real[:] t):
    _surface3d.translate_surface3d(gr.ptr, &t[0])