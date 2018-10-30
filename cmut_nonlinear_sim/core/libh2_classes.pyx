

from . libh2 cimport *


cdef class Vector:

    def __cinit__(self):
        self.ptr = NULL
        self.owner = False

    def __init__(self, uint dim):
        cdef pavector ptr = new_avector(dim)
        self._setup(ptr, True)

    def __dealloc(self):
        if self.ptr is not NULL and self.owner is True:
            del_avector(self.ptr)

    cdef _setup(self, pavector ptr, bint owner):
        self.ptr = ptr
        self.owner = owner
        self.v = <field [:ptr.dim]> (<field *> ptr.v)

    @property
    def dim(self):
        return self.ptr.dim

    @staticmethod
    cdef wrap(pavector ptr, bint owner=False):
        cdef Vector obj = Vector.__new__(Vector)
        obj._setup(ptr, owner)
        return obj


cdef class Matrix():

    def __cinit__(self):
        self.ptr = NULL
        self.owner = False

    def __init__(self, uint rows, uint cols):
        cdef pamatrix ptr = new_amatrix(rows, cols)
        self._setup(ptr, True)

    def __dealloc(self):
        if self.ptr is not NULL and self.owner is True:
            del_amatrix(self.ptr)

    cdef _setup(self, pamatrix ptr, bint owner):
        self.ptr = ptr
        self.owner = owner
        self.a = <field [:ptr.rows,:ptr.cols]> (<field *> ptr.a)

    @property
    def rows(self):
        return self.ptr.rows

    @property
    def cols(self):
        return self.ptr.cols

    @staticmethod
    cdef wrap(pamatrix ptr, bint owner=False):
        cdef Matrix obj = Matrix.__new__(Matrix)
        obj._setup(ptr, owner)
        return obj


cdef class Macrosurface():

    def __cinit__(self):
        self.ptr = NULL
        self.owner = False

    def __init__(self, uint vertices, uint edges, uint triangles):
        cdef pmacrosurface3d ptr = new_macrosurface3d(vertices, edges, triangles)
        self._setup(ptr, True)

    def __dealloc(self):
        if self.ptr is not NULL and self.owner is True:
            del_macrosurface3d(self.ptr)

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
        cdef Macrosurface obj = Macrosurface.__new__(Macrosurface)
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


cdef class Surface:

    def __cinit__(self):
        self.ptr = NULL
        self.owner = False

    def __init__(self, uint vertices, uint edges, uint triangles):
        cdef psurface3d ptr = new_surface3d(vertices, edges, triangles)
        self._setup(ptr, True)

    def __dealloc(self):
        if self.ptr is not NULL and self.owner is True:
            del_surface3d(self.ptr)

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
        cdef Surface obj = Surface.__new__(Surface)
        obj._setup(ptr, owner)
        return obj


cdef class Bem:

    def __cinit__(self):
        self.ptr = NULL
        self.owner = False

    def __init__(self, Surface surf, basisfunctionbem3d row_basis, basisfunctionbem3d col_basis):
        cdef pbem3d ptr = new_bem3d(<pcsurface3d> surf.ptr, row_basis, col_basis)
        self._setup(ptr, True)

    def __dealloc(self):
        if self.ptr is not NULL and self.owner is True:
            del_bem3d(self.ptr)

    cdef _setup(self, pbem3d ptr, bint owner):
        if ptr is NULL:
            raise MemoryError
        self.ptr = ptr
        self.owner = owner

    @property
    def k(self):
        return self.ptr.k

    # @k.setter
    # def k(self, val):
        # self.ptr.k = val

    @property
    def kernel_const(self):
        return self.ptr.kernel_const

    # @kernel_const.setter
    # def kernel_const(self, val):
        # self.ptr.kernel_const = val

    # @property
    # def row_basis(self):
        # return self.ptr.row_basis

    # @row_basis.setter
    # def row_basis(self, val):
        # self.ptr.row_basis = val

    # @property
    # def col_basis(self):
        # return self.ptr.col_basis

    # @col_basis.setter
    # def col_basis(self, val):
        # self.ptr.col_basis = val

    @staticmethod
    cdef wrap(pbem3d ptr, bint owner=False):
        cdef Bem obj = Bem.__new__(Bem)
        obj._setup(ptr, owner)
        return obj


cdef class Cluster:

    def __cinit__(self):
        self.ptr = NULL
        self.owner = False

    def __init__(self, uint size, uint [::1] idx, uint sons, uint dim):
        cdef pcluster ptr = new_cluster(size, &idx[0], sons, dim)
        self._setup(ptr, True)

    def __dealloc(self):
        if self.ptr is not NULL and self.owner is True:
            del_cluster(self.ptr)
    
    cdef _setup(self, pcluster ptr, bint owner):
        self.ptr = ptr
        self.owner = owner
        self.idx = <uint [:ptr.size]> ptr.idx
        self.bmin = <real [:ptr.size]> ptr.bmin
        self.bmax = <real [:ptr.size]> ptr.bmax

    @property
    def size(self):
        return self.ptr.size

    @property
    def sons(self):
        return self.ptr.sons

    @property
    def dim(self):
        return self.ptr.dim

    @property
    def desc(self):
        return self.ptr.desc

    @property
    def type(self):
        return self.ptr.type

    @staticmethod
    cdef wrap(pcluster ptr, bint owner=False):
        cdef Cluster obj = Cluster.__new__(Cluster)
        obj._setup(ptr, owner)
        return obj


cdef class Block:

    def __cinit__(self):
        self.ptr = NULL
        self.owner = False

    def __init__(self, Cluster rc, Cluster cc, bint a, uint rsons, uint csons):
        cdef pblock ptr = new_block(rc.ptr, cc.ptr, a, rsons, csons)
        self._setup(ptr, True)

    def __dealloc(self):
        if self.ptr is not NULL and self.owner is True:
            del_block(self.ptr)

    cdef _setup(self, pblock ptr, bint owner):
        self.ptr = ptr
        self.owner = owner

    @property
    def a(self):
        return self.ptr.a

    @property
    def rsons(self):
        return self.ptr.rsons

    @property
    def csons(self):
        return self.ptr.csons

    @property
    def desc(self):
        return self.ptr.desc

    @staticmethod
    cdef wrap(pblock ptr, bint owner=False):
        cdef Block obj = Block.__new__(Block)
        obj._setup(ptr, owner)
        return obj


cdef class RKMatrix:

    def __cinit__(self):
        self.ptr = NULL
        self.owner = False

    def __init__(self, uint rows, uint cols, uint k):
        cdef prkmatrix ptr = new_rkmatrix(rows, cols, k)
        self._setup(ptr, True)

    def __dealloc(self):
        if self.ptr is not NULL and self.owner is True:
            del_rkmatrix(self.ptr)

    cdef _setup(self, prkmatrix ptr, bint owner):
        self.ptr = ptr
        self.owner = owner

    @property
    def k(self):
        return self.ptr.k

    @staticmethod
    cdef wrap(prkmatrix ptr, bint owner=False):
        cdef RKMatrix obj = RKMatrix.__new__(RKMatrix)
        obj._setup(ptr, owner)
        return obj


cdef class HMatrix:

    def __cinit__(self):
        self.ptr = NULL
        self.owner = False

    def __init__(self, Cluster rc, Cluster cc):
        cdef phmatrix ptr = new_hmatrix(rc.ptr, cc.ptr)
        self._setup(ptr, owner=True)

    def __dealloc(self):
        if self.ptr is not NULL and self.owner is True:
            del_hmatrix(self.ptr)

    cdef _setup(self, phmatrix ptr, bint owner):
        self.ptr = ptr
        self.owner = owner

    @property
    def rsons(self):
        return self.ptr.rsons

    @property
    def csons(self):
        return self.ptr.csons

    @property
    def desc(self):
        return self.ptr.desc

    @staticmethod
    cdef wrap(phmatrix ptr, bint owner=False):
        cdef HMatrix obj = HMatrix.__new__(HMatrix)
        obj._setup(ptr, owner)
        return obj


