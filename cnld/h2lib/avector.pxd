## avector_cy.pxd ##


from . cimport _avector
from . basic cimport *
# cimport numpy as np

ctypedef _avector.pavector pavector
ctypedef _avector.pcavector pcavector

cdef class AVector:
    cdef pavector ptr
    cdef bint owner
    cdef public field [:] v
    cdef _setup(self, pavector ptr, bint owner)
    @staticmethod
    cdef wrap(pavector ptr, bint owner=*)

cpdef random_avector(AVector v)
cpdef clear_avector(AVector v)
cpdef AVector new_zero_avector(uint dim)
cpdef copy_avector(AVector v, AVector w)