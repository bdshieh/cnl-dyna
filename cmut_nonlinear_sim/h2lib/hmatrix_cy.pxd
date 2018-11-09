## hmatrix_cy.pxd ##


from . cimport hmatrix as _hmatrix
from . basic_cy cimport *
from . amatrix_cy cimport *
from . cluster_cy cimport *
from . block_cy cimport *
from . rkmatrix_cy cimport *


ctypedef _hmatrix.phmatrix phmatrix
ctypedef _hmatrix.pchmatrix pchmatrix

cdef class HMatrix:
    cdef phmatrix ptr
    cdef bint owner
    cdef _setup(self, phmatrix ptr, bint owner)
    @staticmethod
    cdef wrap(phmatrix ptr, bint owner=*)

cpdef HMatrix build_from_block_hmatrix(Block b, uint k)