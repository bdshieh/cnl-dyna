## sparsematrix.pxd ##


from . cimport sparsepattern as _sparsepattern
from . basic_cy cimport *


ctypedef _sparsepattern.psparsepattern psparsepattern
ctypedef _sparsepattern.pcsparsepattern pcsparsepattern

cdef class SparsePattern:
    cdef psparsepattern ptr
    cdef bint owner
    cdef _setup(self, psparsepattern ptr, bint owner)
    @staticmethod
    cdef wrap(psparsepattern ptr, bint owner=*)

cpdef SparsePattern new_sparsepattern(uint rows, uint cols)
cpdef del_sparsepattern(SparsePattern sp)
cpdef clear_sparsepattern(SparsePattern sp)
cpdef addnz_sparsepattern(SparsePattern sp, uint row, uint col)