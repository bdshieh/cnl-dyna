## rkmatrix_cy.pyx ##


from . cimport _rkmatrix 
from . basic cimport *
from . amatrix cimport *


cdef class RKMatrix:

    ''' Initialization methods '''
    def __cinit__(self):
        self.ptr = NULL
        self.owner = False

    def __init__(self, uint rows, uint cols, uint k):
        cdef prkmatrix ptr = _rkmatrix.new_rkmatrix(rows, cols, k)
        self._setup(ptr, True)

    def __dealloc__(self):
        if self.ptr is not NULL and self.owner is True:
            _rkmatrix.del_rkmatrix(self.ptr)

    cdef _setup(self, prkmatrix ptr, bint owner):
        self.ptr = ptr
        self.owner = owner

    ''' Scalar properties '''
    @property
    def k(self):
        return self.ptr.k

    ''' Pointer properties '''
    @property
    def A(self):
        return AMatrix.wrap(&self.ptr.A, False)

    @property
    def B(self):
        return AMatrix.wrap(&self.ptr.B, False)

    ''' Methods '''
    @staticmethod
    cdef wrap(prkmatrix ptr, bint owner=True):
        cdef RKMatrix obj = RKMatrix.__new__(RKMatrix)
        obj._setup(ptr, owner)
        return obj