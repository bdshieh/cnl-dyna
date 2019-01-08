## macrosurface3d_cy.pyx ##


from . cimport _macrosurface3d
from . basic cimport *
from . surface3d cimport *


cdef class Macrosurface3d():

    def __cinit__(self):
        self.ptr = NULL
        self.owner = False

    def __init__(self, uint vertices, uint edges, uint triangles):
        cdef pmacrosurface3d ptr = _macrosurface3d.new_macrosurface3d(vertices, edges, triangles)
        self._setup(ptr, True)

    def __dealloc__(self):
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
    
    def set_parametrization(self, str type):

        if type.lower() in ['square']:
            self.ptr.phi = cube_parametrization
        elif type.lower() in ['circle']:
            self.ptr.phi = circle_parametrization   
        elif type.lower() in ['cube']:
            self.ptr.phi = cube_parametrization        
        elif type.lower() in ['sphere']:
            self.ptr.phi = sphere_parametrization
        elif type.lower() in ['cylinder']:
            self.ptr.phi = cylinder_parametrization
        else:
            raise TypeError

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

    assert(i < mg.triangles)
    assert(t[i][0] < mg.vertices)
    assert(t[i][1] < mg.vertices)
    assert(t[i][2] < mg.vertices)

    xt[0] = (x[t[i][0]][0] * (1.0 - xr1 - xr2) + x[t[i][1]][0] * xr1 + x[t[i][2]][0] * xr2)
    xt[1] = (x[t[i][0]][1] * (1.0 - xr1 - xr2) + x[t[i][1]][1] * xr1 + x[t[i][2]][1] * xr2)
    xt[2] = (x[t[i][0]][2] * (1.0 - xr1 - xr2) + x[t[i][1]][2] * xr1 + x[t[i][2]][2] * xr2)


cdef void sphere_parametrization(uint i, real xr1, real xr2, void * data, real xt[3]):

    cdef pcmacrosurface3d mg = <pcmacrosurface3d> data
    cdef real [:,:] x = <real [:mg.vertices,:3]> (<real *> mg.x)
    cdef uint [:,:] t = <uint [:mg.triangles,:3]> (<uint *> mg.t)
    cdef real norm

    assert(i < mg.triangles)
    assert(t[i][0] < mg.vertices)
    assert(t[i][1] < mg.vertices)
    assert(t[i][2] < mg.vertices)

    xt[0] = (x[t[i][0]][0] * (1.0 - xr1 - xr2) + x[t[i][1]][0] * xr1 + x[t[i][2]][0] * xr2)
    xt[1] = (x[t[i][0]][1] * (1.0 - xr1 - xr2) + x[t[i][1]][1] * xr1 + x[t[i][2]][1] * xr2)
    xt[2] = (x[t[i][0]][2] * (1.0 - xr1 - xr2) + x[t[i][1]][2] * xr1 + x[t[i][2]][2] * xr2)

    norm = _macrosurface3d.REAL_NORM3(xt[0], xt[1], xt[2])
    xt[0] /= norm
    xt[1] /= norm
    xt[2] /= norm


cdef void circle_parametrization(uint i, real xr1, real xr2, void * data, real xt[3]):

    cdef pcmacrosurface3d mg = <pcmacrosurface3d> data
    cdef real [:,:] x = <real [:mg.vertices,:3]> (<real *> mg.x)
    cdef uint [:,:] t = <uint [:mg.triangles,:3]> (<uint *> mg.t)
    cdef real normL1, normL2

    assert(i < mg.triangles)
    assert(t[i][0] < mg.vertices)
    assert(t[i][1] < mg.vertices)
    assert(t[i][2] < mg.vertices)

    xt[0] = (x[t[i][0]][0] * (1.0 - xr1 - xr2) + x[t[i][1]][0] * xr1 + x[t[i][2]][0] * xr2)
    xt[1] = (x[t[i][0]][1] * (1.0 - xr1 - xr2) + x[t[i][1]][1] * xr1 + x[t[i][2]][1] * xr2)
    xt[2] = (x[t[i][0]][2] * (1.0 - xr1 - xr2) + x[t[i][1]][2] * xr1 + x[t[i][2]][2] * xr2)

    normL1 = _macrosurface3d.REAL_ABS(xt[0]) + _macrosurface3d.REAL_ABS(xt[1])
    normL2 = _macrosurface3d.REAL_NORM2(xt[0], xt[1])
    if (normL2 > 0.0):
        xt[0] = xt[0] * normL1 / normL2
        xt[1] = xt[1] * normL1 / normL2


cdef void cylinder_parametrization(uint i, real xr1, real xr2, void * data, real xt[3]):

    cdef pcmacrosurface3d mg = <pcmacrosurface3d> data
    cdef real [:,:] x = <real [:mg.vertices,:3]> (<real *> mg.x)
    cdef uint [:,:] t = <uint [:mg.triangles,:3]> (<uint *> mg.t)
    cdef real normL1, normL2

    assert(i < mg.triangles)
    assert(t[i][0] < mg.vertices)
    assert(t[i][1] < mg.vertices)
    assert(t[i][2] < mg.vertices)

    xt[0] = (x[t[i][0]][0] * (1.0 - xr1 - xr2) + x[t[i][1]][0] * xr1 + x[t[i][2]][0] * xr2)
    xt[1] = (x[t[i][0]][1] * (1.0 - xr1 - xr2) + x[t[i][1]][1] * xr1 + x[t[i][2]][1] * xr2)
    xt[2] = (x[t[i][0]][2] * (1.0 - xr1 - xr2) + x[t[i][1]][2] * xr1 + x[t[i][2]][2] * xr2)

    if (_macrosurface3d.REAL_ABS(xt[0]) != 4.0):
        normL2 = _macrosurface3d.REAL_NORM2(xt[1], xt[2])
        xt[1] /= normL2
        xt[2] /= normL2
    else:
        normL1 = _macrosurface3d.REAL_ABS(xt[1]) + _macrosurface3d.REAL_ABS(xt[2])
        normL2 = _macrosurface3d.REAL_NORM2(xt[1], xt[2])
        if (normL2 > 0.0):
            xt[1] = xt[1] * normL1 / normL2
            xt[2] = xt[2] * normL1 / normL2


cpdef build_from_macrosurface3d_surface3d(Macrosurface3d ms, uint refn):

    cdef psurface3d surf = _macrosurface3d.build_from_macrosurface3d_surface3d(<pcmacrosurface3d> ms.ptr, refn)
    return Surface3d.wrap(surf, True)

cpdef Macrosurface3d new_sphere_macrosurface3d():

    cdef pmacrosurface3d ms = _macrosurface3d.new_sphere_macrosurface3d()
    return Macrosurface3d.wrap(ms, True)