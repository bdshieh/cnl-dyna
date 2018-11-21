## sparsematrix_cy.pyx ##


from . cimport sparsematrix as _sparsematrix
from . basic_cy cimport *
import numpy as np
import scipy as sp
import scipy.sparse

cdef class SparseMatrix():

    def __cinit__(self):
        self.ptr = NULL
        self.owner = False

    def __init__(self, uint rows, uint cols):
        cdef psparsematrix ptr = _sparsematrix.new_sparsematrix(rows, cols)
        self._setup(ptr, True)

    @classmethod
    def from_array(cls, a):

        a = a.squeeze()
        assert a.ndim == 2

        if not isinstance(a, sp.sparse.csr_matrix):
            a = sp.sparse.csr_matrix(a)

        obj = cls(a.shape[0], a.shape[1])
        obj.rows = 
        obj.cols
        obj.coeffs
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
    pass

cpdef SparseMatrix new_identity_sparsematrix(uint rows, uint cols):
    pass

cpdef field addentry_sparsematrix(SparseMatrix a, uint row, uint col, field x):
    pass

cpdef setentry_sparsematrix(SparseMatrix a, uint row, uint col, field x):
    pass
    
cpdef size_t getsize_sparsematrix(SparseMatrix a):
    pass

cpdef sort_sparsematrix(SparseMatrix a):
    pass

cpdef clear_sparsematrix(SparseMatrix a):
    pass





