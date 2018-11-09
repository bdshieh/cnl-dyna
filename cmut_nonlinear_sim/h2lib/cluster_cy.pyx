## cluster_cy.pyx ##


from . cimport cluster as _cluster
from . basic_cy cimport *


cdef class Cluster:

    def __cinit__(self):
        self.ptr = NULL
        self.owner = False

    def __init__(self, uint size, uint [::1] idx, uint sons, uint dim):
        cdef pcluster ptr = _cluster.new_cluster(size, &idx[0], sons, dim)
        self._setup(ptr, True)

    def __dealloc(self):
        if self.ptr is not NULL and self.owner is True:
            _cluster.del_cluster(self.ptr)
    
    cdef _setup(self, pcluster ptr, bint owner):
        self.ptr = ptr
        self.owner = owner
        self.idx = <uint [:ptr.size]> (<uint *> ptr.idx)
        self.bmin = <real [:ptr.size]> (<real *> ptr.bmin)
        self.bmax = <real [:ptr.size]> (<real *> ptr.bmax)

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