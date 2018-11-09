## surface3d_cy.pyx ##


from . cimport surface3d as _surface3d
from . basic_cy cimport *


cdef class Surface3d():

    def __cinit__(self):
        self.ptr = NULL
        self.owner = False

    def __init__(self, uint vertices, uint edges, uint triangles):
        cdef psurface3d ptr = _surface3d.new_surface3d(vertices, edges, triangles)
        self._setup(ptr, True)

    def __dealloc(self):
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

    @property
    def vertices(self):
        return self.ptr.vertices

    @property
    def edges(self):
        return self.ptr.edges

    @property
    def triangles(self):
        return self.ptr.triangles

    @staticmethod
    cdef wrap(psurface3d ptr, bint owner=False):
        cdef Surface3d obj = Surface3d.__new__(Surface3d)
        obj._setup(ptr, owner)
        return obj