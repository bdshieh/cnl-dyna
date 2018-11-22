## sparsepattern_cy.pyx ##


from . cimport sparsepattern as _sparsepattern
from . basic_cy cimport *


cdef class SparsePattern:

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

    @property
    def rows(self):
        return self.ptr.rows

    @property
    def cols(self):
        return self.ptr.cols

    @property
    def row(self):
        return PatEntry.wrap(self.ptr.row[0], False)

    @staticmethod
    cdef wrap(psparsepattern ptr, bint owner=False):
        cdef SparsePattern obj = SparsePattern.__new__(SparsePattern)
        obj._setup(ptr, owner)
        return obj

cdef class PatEntry:

    def __cinit__(self):
        self.ptr = NULL
        self.owner = False

    def __init__(self):
        pass

    def __dealloc(self):
        pass

    cdef _setup(self, ppatentry ptr, bint owner):
        self.ptr = ptr
        self.owner = owner

    @property
    def row(self):
        return self.ptr.row

    @property
    def col(self):
        return self.ptr.col
    
    @property
    def next(self):
        return PatEntry.wrap(self.ptr.next[0], False)

    @staticmethod
    cdef wrap(ppatentry ptr, bint owner=False):
        cdef PatEntry obj = PatEntry.__new__(PatEntry)
        obj._setup(ptr, owner)
        return obj

cpdef SparsePattern new_sparsepattern(uint rows, uint cols):
    cdef sp = _sparsepattern.new_sparsepattern(rows, cols)
    return SparsePattern.wrap(sp, True)

cpdef del_sparsepattern(SparsePattern sp):
    _sparsepattern.del_sparsepattern(sp.ptr)

cpdef clear_sparsepattern(SparsePattern sp):
    _sparsepattern.clear_sparsepattern(sp.ptr)

cpdef addnz_sparsepattern(SparsePattern sp, uint row, uint col):
    _sparsepattern.addnz_sparsepattern(sp.ptr, row, col)


