## truncation_cy.pyx ##


from . cimport _truncation
from . basic cimport *


cdef class Truncmode:

    def __cinit__(self):
        self.ptr = NULL
        self.owner = False

    def __init__(self):
        cdef ptruncmode ptr = _truncation.new_truncmode()
        self._setup(ptr, owner=True)

    def __dealloc__(self):
        if self.ptr is not NULL and self.owner is True:
            _truncation.del_truncmode(self.ptr)

    cdef _setup(self, ptruncmode ptr, bint owner):
        self.ptr = ptr
        self.owner = owner

    @property
    def frobenius(self):
        return self.ptr.frobenius

    @frobenius.setter
    def frobenius(self, val):
        self.ptr.frobenius = val

    @property
    def absolute(self):
        return self.ptr.absolute

    @absolute.setter
    def absolute(self, val):
        self.ptr.absolute = val

    @property
    def blocks(self):
        return self.ptr.blocks

    @blocks.setter
    def blocks(self, val):
        self.ptr.blocks = val

    @property
    def zeta_level(self):
        return self.ptr.zeta_level

    @zeta_level.setter
    def zeta_level(self, val):
        self.ptr.zeta_level = val

    @property
    def zeta_age(self):
        return self.ptr.zeta_age

    @zeta_age.setter
    def zeta_age(self, val):
        self.ptr.zeta_age = val

    @staticmethod
    cdef wrap(ptruncmode ptr, bint owner=True):
        cdef Truncmode obj = Truncmode.__new__(Truncmode)
        obj._setup(ptr, owner)
        return obj

cpdef Truncmode new_releucl_truncmode():
    cdef ptruncmode tm = _truncation.new_releucl_truncmode()
    return Truncmode.wrap(tm, True)