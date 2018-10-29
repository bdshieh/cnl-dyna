
from . libh2 cimport *


cdef class Vector:

    def __cinit__(self):
        self.ptr = NULL
        self.owner = False

    def __dealloc(self):
        if self.ptr is not NULL and self.owner is True:
            del_avector(self.ptr)
    
    @property
    def dim(self):
        return self.ptr.dim

    @staticmethod
    cdef create(uint dim):

        cdef pavector ptr = new_avector(dim)
        return Vector.wrap(ptr, owner=True)

    @staticmethod
    cdef wrap(pavector ptr, bint owner=False):

        cdef Vector obj = Vector()
        obj.ptr = ptr
        obj.owner = owner
        obj.v = <field [:ptr.dim]> ptr.v
        return obj


cdef class Matrix():

    def __cinit__(self):
        self.ptr = NULL
        self.owner = False

    def __dealloc(self):
        if self.ptr is not NULL and self.owner is True:
            del_amatrix(self.ptr)

    @property
    def rows(self):
        return self.ptr.rows

    @property
    def cols(self):
        return self.ptr.cols

    @staticmethod
    cdef create(uint rows, uint cols):

        cdef pamatrix ptr = new_amatrix(rows, cols)
        return Matrix.wrap(ptr, owner=True)

    @staticmethod
    cdef wrap(pamatrix ptr, bint owner=False):

        cdef Matrix obj = Matrix()
        obj.ptr = ptr
        obj.owner = owner
        obj.a = <field [:ptr.rows,:ptr.cols]> ptr.a
        return obj


cdef class Macrosurface():

    def __cinit__(self):
        self.ptr = NULL
        self.owner = False

    def __dealloc(self):
        if self.ptr is not NULL and self.owner is True:
            del_macrosurface3d(self.ptr)

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
    cdef create(uint vertices, uint edges, uint triangles):

        cdef pmacrosurface3d ptr = new_macrosurface3d(vertices, edges, triangles)
        ptr.phi = NULL
        ptr.phidata = ptr
        return Macrosurface.wrap(ptr, owner=True)

    @staticmethod
    cdef wrap(pmacrosurface3d ptr, bint owner=False):

        cdef Macrosurface obj = Macrosurface()
        obj.ptr = ptr
        obj.owner = owner
        obj.x = <real [:ptr.vertices,:3]> (<real *> ptr.x)
        obj.e = <uint [:ptr.edges,:2]> (<uint *> ptr.e)
        obj.t = <uint [:ptr.triangles,:3]> (<uint *> ptr.t)
        obj.s = <uint [:ptr.triangles,:3]> (<uint *> ptr.s)

        return obj


cdef class Surface:

    def __cinit__(self):
        self.ptr = NULL
        self.owner = False

    def __dealloc(self):
        if self.ptr is not NULL and self.owner is True:
            del_surface3d(self.ptr)

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
    cdef create(uint vertices, uint edges, uint triangles):

        cdef psurface3d ptr = new_surface3d(vertices, edges, triangles)
        return Surface.wrap(ptr, owner=True)

    @staticmethod
    cdef wrap(psurface3d ptr, bint owner=False):

        cdef Surface obj = Surface()
        obj.ptr = ptr
        obj.owner = owner
        obj.x = <real [:ptr.vertices,:3]> (<real *> ptr.x)
        obj.e = <uint [:ptr.edges,:2]> (<uint *> ptr.e)
        obj.t = <uint [:ptr.triangles,:3]> (<uint *> ptr.t)
        obj.s = <uint [:ptr.triangles,:3]> (<uint *> ptr.s)
        obj.n = <real [:ptr.triangles,:3]> (<real *> ptr.n)

        return obj


cdef class Bem:

    def __cinit__(self):
        self.ptr = NULL
        self.owner = False

    def __dealloc(self):
        if self.ptr is not NULL and self.owner is True:
            del_bem3d(self.ptr)

    @property
    def k(self):
        return self.ptr.k

    @k.setter
    def k(self, val):
        self.ptr.k = val

    @property
    def kernel_const(self):
        return self.ptr.kernel_const

    @kernel_const.setter
    def kernel_const(self, val):
        self.ptr.kernel_const = val

    @property
    def row_basis(self):
        return self.ptr.row_basis

    @row_basis.setter
    def row_basis(self, val):
        self.ptr.row_basis = val

    @property
    def col_basis(self):
        return self.ptr.col_basis

    @col_basis.setter
    def col_basis(self, val):
        self.ptr.col_basis = val

    @staticmethod
    cdef create(pcsurface3d gr, basisfunctionbem3d row_basis, basisfunctionbem3d col_basis):

        cdef pbem3d ptr = new_bem3d(gr, row_basis, col_basis)
        return Bem.wrap(ptr, owner=True)

    @staticmethod
    cdef wrap(pbem3d ptr, bint owner=False):

        cdef Bem obj = Bem()
        obj.ptr = ptr
        obj.owner = owner
        return obj


cdef class Cluster:

    def __cinit__(self):
        self.ptr = NULL
        self.owner = False

    def __dealloc(self):
        if self.ptr is not NULL and self.owner is True:
            del_cluster(self.ptr)

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
    cdef create(uint size, uint * idx, uint sons, uint dim):

        cdef pcluster ptr = new_cluster(size, idx, sons, dim)
        return Cluster.wrap(ptr, owner=True)

    @staticmethod
    cdef wrap(pcluster ptr, bint owner=False):

        cdef Cluster obj = Cluster()
        obj.ptr = ptr
        obj.owner = owner
        obj.idx = <uint [:ptr.size]> ptr.idx
        obj.bmin = <real [:ptr.size]> ptr.bmin
        obj.bmax = <real [:ptr.size]> ptr.bmax
        return obj


cdef class Block:

    def __cinit__(self):
        self.ptr = NULL
        self.owner = False

    def __dealloc(self):
        if self.ptr is not NULL and self.owner is True:
            del_block(self.ptr)

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
    cdef create(pcluster rc, pcluster cc, bint a, uint rsons, uint csons):

        cdef pblock ptr = new_block(rc, cc, a, rsons, csons)
        return Block.wrap(ptr, owner=True)

    @staticmethod
    cdef wrap(pblock ptr, bint owner=False):

        cdef Block obj = Block()
        obj.ptr = ptr
        obj.owner = owner
        return obj


cdef class RKMatrix:

    def __cinit__(self):
        self.ptr = NULL
        self.owner = False

    def __dealloc(self):
        if self.ptr is not NULL and self.owner is True:
            del_rkmatrix(self.ptr)

    @property
    def k(self):
        return self.ptr.k

    @staticmethod
    cdef create(uint rows, uint cols, uint k):

        cdef prkmatrix ptr = new_rkmatrix(rows, cols, k)
        return RKMatrix.wrap(ptr, owner=True)

    @staticmethod
    cdef wrap(prkmatrix ptr, bint owner=False):

        cdef RKMatrix obj = RKMatrix()
        obj.ptr = ptr
        obj.owner = owner
        return obj


cdef class HMatrix:

    def __cinit__(self):
        self.ptr = NULL
        self.owner = False

    def __dealloc(self):
        if self.ptr is not NULL and self.owner is True:
            del_hmatrix(self.ptr)

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
    cdef create(pcluster rc, pcluster cc):

        cdef phmatrix ptr = new_hmatrix(rc, cc)
        return HMatrix.wrap(ptr, owner=True)

    @staticmethod
    cdef wrap(phmatrix ptr, bint owner=False):

        cdef HMatrix obj = HMatrix()
        obj.ptr = ptr
        obj.owner = owner
        return obj


