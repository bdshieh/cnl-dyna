## macrosurface3d_cy.pyx ##


from . cimport macrosurface3d as _macrosurface3d
from . basic_cy cimport *
from . surface3d_cy cimport *


cdef class Macrosurface3d():

    def __cinit__(self):
        self.ptr = NULL
        self.owner = False

    def __init__(self, uint vertices, uint edges, uint triangles):
        cdef pmacrosurface3d ptr = _macrosurface3d.new_macrosurface3d(vertices, edges, triangles)
        self._setup(ptr, True)

    def __dealloc(self):
        if self.ptr is not NULL and self.owner is True:
            _macrosurface3d.del_macrosurface3d(self.ptr)

    cdef _setup(self, pmacrosurface3d ptr, bint owner):
        self.ptr = ptr
        self.owner = owner
        self.x = <real [:ptr.vertices,:3]> (<real *> ptr.x)
        self.e = <uint [:ptr.edges,:2]> (<uint *> ptr.e)
        self.t = <uint [:ptr.triangles,:3]> (<uint *> ptr.t)
        self.s = <uint [:ptr.triangles,:3]> (<uint *> ptr.s)
        ptr.phi = cube_parametrization
        ptr.phidata = ptr

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
    cdef wrap(pmacrosurface3d ptr, bint owner=False):
        cdef Macrosurface3d obj = Macrosurface3d.__new__(Macrosurface3d)
        obj._setup(ptr, owner)
        return obj


cdef void cube_parametrization(uint i, real xr1, real xr2, void * data, real xt[3]):

    cdef pmacrosurface3d mg = <pmacrosurface3d> data
    cdef real [:,:] x = <real [:mg.vertices,:3]> (<real *> mg.x)
    cdef uint [:,:] t = <uint [:mg.triangles,:3]> (<uint *> mg.t)
    # cdef const real(* x)[3] = <const real(*)[3]> mg.x
    # cdef const uint(* t)[3] = <const uint(*)[3]> mg.t

    assert(i < mg.triangles)
    assert(t[i][0] < mg.vertices)
    assert(t[i][1] < mg.vertices)
    assert(t[i][2] < mg.vertices)

    xt[0] = (x[t[i][0]][0] * (1.0 - xr1 - xr2) + x[t[i][1]][0] * xr1 + x[t[i][2]][0] * xr2)
    xt[1] = (x[t[i][0]][1] * (1.0 - xr1 - xr2) + x[t[i][1]][1] * xr1 + x[t[i][2]][1] * xr2)
    xt[2] = (x[t[i][0]][2] * (1.0 - xr1 - xr2) + x[t[i][1]][2] * xr1 + x[t[i][2]][2] * xr2)


cpdef build_from_macrosurface3d_surface3d(Macrosurface3d ms, uint refn):

    cdef psurface3d surf = _macrosurface3d.build_from_macrosurface3d_surface3d(<pcmacrosurface3d> ms.ptr, refn)
    return Surface3d.wrap(surf, True)