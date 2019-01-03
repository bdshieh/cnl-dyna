## hmatrix_cy.pxd ##


from . cimport _hmatrix
from . basic cimport *
from . amatrix cimport *
from . cluster cimport *
from . block cimport *
from . rkmatrix cimport *
from . sparsematrix cimport *


ctypedef _hmatrix.phmatrix phmatrix
ctypedef _hmatrix.pchmatrix pchmatrix

cdef class HMatrix:
    cdef phmatrix ptr
    cdef bint owner
    cdef _setup(self, phmatrix ptr, bint owner)
    @staticmethod
    cdef wrap(phmatrix ptr, bint owner=*)

cpdef HMatrix build_from_block_hmatrix(Block b, uint k)
cpdef clear_hmatrix(HMatrix hm)
cpdef copy_hmatrix(HMatrix src, HMatrix trg)
cpdef HMatrix clone_hmatrix(HMatrix src)
cpdef size_t getsize_hmatrix(HMatrix hm)
cpdef addeval_hmatrix_avector(field alpha, HMatrix hm, AVector x, AVector y)
cpdef addevalsymm_hmatrix_avector(field alpha, HMatrix hm, AVector x, AVector y)
cpdef uint getrows_hmatrix(HMatrix hm)
cpdef uint getcols_hmatrix(HMatrix hm)
cpdef HMatrix clonestructure_hmatrix(HMatrix src)
cpdef copy_sparsematrix_hmatrix(SparseMatrix sp, HMatrix hm)
cpdef identity_hmatrix(HMatrix hm)