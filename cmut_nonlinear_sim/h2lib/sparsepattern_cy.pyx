## sparsematrix_cy.pyx ##


from . cimport sparsepattern as _sparsepattern
from . basic_cy cimport *
import numpy as np
import scipy as sp
import scipy.sparse

cdef class SparsePattern():

    def __cinit__(self):
        self.ptr = NULL
        self.owner = False

    def __init__(self, uint rows, uint cols):
        cdef psparsepattern ptr = _sparsepattern.new_sparsepattern(rows, cols)
        self._setup(ptr, True)

    def __dealloc(self):
        if self.ptr is not NULL and self.owner is True:
            _sparsepattern.del_sparsepattern(self.ptr)

    cdef _setup(self, psparsepattern ptr, bint owner):
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
    def row(self):
        return np.asarray(self._row)

    @staticmethod
    cdef wrap(psparsepattern ptr, bint owner=False):
        cdef SparsePattern obj = SparsePattern.__new__(SparsePattern)
        obj._setup(ptr, owner)
        return obj

cpdef SparsePattern new_sparsepattern(uint rows, uint cols):


cpdef del_sparsepattern(SparsePattern sp):


cpdef clear_sparsepattern(SparsePattern sp):


cpdef addnz_sparsepattern(SparsePattern sp, uint row, uint col):

