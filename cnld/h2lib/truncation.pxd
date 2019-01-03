## truncation_cy.pxd ##


from . cimport _truncation
from . basic cimport *


ctypedef _truncation.ptruncmode ptruncmode
ctypedef _truncation.pctruncmode pctruncmode

cdef class Truncmode:
    cdef ptruncmode ptr
    cdef bint owner
    cdef _setup(self, ptruncmode ptr, bint owner)
    @staticmethod
    cdef wrap(ptruncmode ptr, bint owner=*)

cpdef Truncmode new_releucl_truncmode()