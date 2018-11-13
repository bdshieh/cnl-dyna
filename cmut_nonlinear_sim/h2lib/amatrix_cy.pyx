## amatrix_cy.pyx ##


from . cimport amatrix as _amatrix
from . basic_cy cimport *
from . avector_cy cimport *
import numpy as np


cdef class AMatrix():

    def __cinit__(self):
        self.ptr = NULL
        self.owner = False

    def __init__(self, uint rows, uint cols):
        cdef pamatrix ptr = _amatrix.new_amatrix(rows, cols)
        self._setup(ptr, True)

    @classmethod
    def from_array(cls, a):

        a = a.squeeze()
        assert a.ndim == 2

        obj = cls(a.shape[0], a.shape[1])
        obj.a[:] = a.astype(np.complex128)
        return obj

    def __dealloc(self):
        if self.ptr is not NULL and self.owner is True:
            _amatrix.del_amatrix(self.ptr)

    cdef _setup(self, pamatrix ptr, bint owner):
        self.ptr = ptr
        self.owner = owner
        self.a = <field [:ptr.rows,:ptr.cols]> (<field *> ptr.a)

    @property
    def rows(self):
        return self.ptr.rows

    @property
    def cols(self):
        return self.ptr.cols

    @staticmethod
    cdef wrap(pamatrix ptr, bint owner=False):
        cdef AMatrix obj = AMatrix.__new__(AMatrix)
        obj._setup(ptr, owner)
        return obj
