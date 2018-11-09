## amatrix_cy.pxd ##


from . cimport amatrix as _amatrix
from . basic_cy cimport *
from . avector_cy cimport *


ctypedef _amatrix.pamatrix pamatrix
ctypedef _amatrix.pcamatrix pcamatrix

cdef class AMatrix:
    cdef pamatrix ptr
    cdef bint owner
    cdef public field [:,:] a
    cdef _setup(self, pamatrix ptr, bint owner)
    @staticmethod
    cdef wrap(pamatrix ptr, bint owner=*)