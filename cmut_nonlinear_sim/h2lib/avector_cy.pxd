## avector_cy.pxd ##


from . cimport avector as _avector
from . basic_cy cimport *


ctypedef _avector.pavector pavector
ctypedef _avector.pcavector pcavector

cdef class AVector:
    cdef pavector ptr
    cdef bint owner
    cdef public field [:] v
    cdef _setup(self, pavector ptr, bint owner)
    @staticmethod
    cdef wrap(pavector ptr, bint owner=*)
