## sparsematrix_cy.pyx ##


from . cimport sparsematrix as _sparsematrix
from . basic_cy cimport *
from . amatrix_cy cimport *
import numpy as np
import scipy as sp
import scipy.sparse


cdef class SparseMatrix():

    def __cinit__(self):
        self.ptr = NULL
        self.owner = False

    def __init__(self, uint rows, uint cols, uint nz):
        cdef psparsematrix ptr = _sparsematrix.new_raw_sparsematrix(rows, cols, nz)
        self._setup(ptr, True)

    @classmethod
    def from_array(cls, a):

        a = a.astype(np.complex128)
        assert a.ndim == 2

        if not isinstance(a, sp.sparse.csr_matrix):
            a = sp.sparse.csr_matrix(a)

        obj = cls(a.shape[0], a.shape[1], a.nnz)
        obj._row[:] = a.indptr
        obj._col[:] = a.indices
        obj._coeff[:] = a.data
        return obj

    def __dealloc(self):
        if self.ptr is not NULL and self.owner is True:
            _sparsematrix.del_sparsematrix(self.ptr)

    cdef _setup(self, psparsematrix ptr, bint owner):
        self.ptr = ptr
        self.owner = owner
        self._row = <uint [:(ptr.rows + 1)]> (<uint *> ptr.row)
        self._col = <uint [:ptr.nz]> (<uint *> ptr.col)
        self._coeff = <field [:ptr.nz]> (<field *> ptr.coeff)

    @property
    def rows(self):
        return self.ptr.rows

    @property
    def cols(self):
        return self.ptr.cols

    @property
    def nz(self):
        return self.ptr.nz

    @property
    def row(self):
        return np.asarray(self._row)

    @property
    def col(self):
        return np.asarray(self._col)

    @property
    def coeff(self):
        return np.asarray(self._coeff)
    
    @staticmethod
    cdef wrap(psparsematrix ptr, bint owner=False):
        cdef SparseMatrix obj = SparseMatrix.__new__(SparseMatrix)
        obj._setup(ptr, owner)
        return obj


cpdef SparseMatrix new_raw_sparsematrix(uint rows, uint cols, uint nz):
    cpdef psparsematrix sm = _sparsematrix.new_raw_sparsematrix(rows, cols, nz)
    return SparseMatrix.wrap(sm, True)

cpdef SparseMatrix new_identity_sparsematrix(uint rows, uint cols):
    cpdef psparsematrix sm = _sparsematrix.new_identity_sparsematrix(rows, cols)
    return SparseMatrix.wrap(sm, True)

cpdef del_sparsematrix(SparseMatrix a):
    _sparsematrix.del_sparsematrix(a.ptr)

cpdef field addentry_sparsematrix(SparseMatrix a, uint row, uint col, field x):
    return _sparsematrix.addentry_sparsematrix(a.ptr, row, col, x)

cpdef setentry_sparsematrix(SparseMatrix a, uint row, uint col, field x):
    _sparsematrix.setentry_sparsematrix(a.ptr, row, col, x)
    
cpdef size_t getsize_sparsematrix(SparseMatrix a):
    return _sparsematrix.getsize_sparsematrix(a.ptr)

cpdef sort_sparsematrix(SparseMatrix a):
    _sparsematrix.sort_sparsematrix(a.ptr)

cpdef clear_sparsematrix(SparseMatrix a):
    _sparsematrix.clear_sparsematrix(a.ptr)

cpdef add_sparsematrix_amatrix(field alpha, bint atrans, SparseMatrix a, AMatrix b):
    _sparsematrix.add_sparsematrix_amatrix(alpha, atrans, <pcsparsematrix> a.ptr, b.ptr)

