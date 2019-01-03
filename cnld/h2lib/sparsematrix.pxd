## sparsematrix.pxd ##


from . cimport _sparsematrix
from . basic cimport *
from . amatrix cimport *


ctypedef _sparsematrix.psparsematrix psparsematrix
ctypedef _sparsematrix.pcsparsematrix pcsparsematrix

cdef class SparseMatrix:
    cdef psparsematrix ptr
    cdef bint owner
    cdef public uint [:] _row
    cdef public uint [:] _col
    cdef public field [:] _coeff
    cdef _setup(self, psparsematrix ptr, bint owner)
    @staticmethod
    cdef wrap(psparsematrix ptr, bint owner=*)

cpdef SparseMatrix new_raw_sparsematrix(uint rows, uint cols, uint nz)
cpdef SparseMatrix new_identity_sparsematrix(uint rows, uint cols)
cpdef del_sparsematrix(SparseMatrix a)
cpdef field addentry_sparsematrix(SparseMatrix a, uint row, uint col, field x)
cpdef setentry_sparsematrix(SparseMatrix a, uint row, uint col, field x)
cpdef size_t getsize_sparsematrix(SparseMatrix a)
cpdef sort_sparsematrix(SparseMatrix a)
cpdef clear_sparsematrix(SparseMatrix a)
cpdef add_sparsematrix_amatrix(field alpha, bint atrans, SparseMatrix a, AMatrix b)
