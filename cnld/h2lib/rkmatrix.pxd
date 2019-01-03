## rkmatrix_cy.pxd ##


from . cimport _rkmatrix 
from . basic cimport *
from . amatrix cimport *


ctypedef _rkmatrix.prkmatrix prkmatrix
ctypedef _rkmatrix.pcrkmatrix pcrkmatrix

cdef class RKMatrix:
    cdef prkmatrix ptr
    cdef bint owner
    cdef _setup(self, prkmatrix ptr, bint owner)
    @staticmethod
    cdef wrap(prkmatrix ptr, bint owner=*)

