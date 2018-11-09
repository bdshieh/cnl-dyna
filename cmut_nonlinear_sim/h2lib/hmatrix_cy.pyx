## hmatrix_cy.pyx ##


from . cimport hmatrix as _hmatrix
from . basic_cy cimport *
from . amatrix_cy cimport *
from . cluster_cy cimport *
from . block_cy cimport *
from . rkmatrix_cy cimport *


cdef class HMatrix:

    def __cinit__(self):
        self.ptr = NULL
        self.owner = False

    def __init__(self, Cluster rc, Cluster cc):
        cdef phmatrix ptr = _hmatrix.new_hmatrix(rc.ptr, cc.ptr)
        self._setup(ptr, owner=True)

    def __dealloc(self):
        if self.ptr is not NULL and self.owner is True:
            _hmatrix.del_hmatrix(self.ptr)

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