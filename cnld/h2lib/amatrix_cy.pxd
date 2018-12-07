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

cpdef AMatrix clone_amatrix(AMatrix src)
cpdef addeval_amatrix_avector(field alpha, AMatrix a, AVector src, AVector trg)
cpdef size_t getsize_amatrix(AMatrix a)
cpdef scale_amatrix(field alpha, AMatrix a)
cpdef conjugate_amatrix(AMatrix a)
cpdef add_amatrix(field alpha, bint atrans, AMatrix a, AMatrix b)