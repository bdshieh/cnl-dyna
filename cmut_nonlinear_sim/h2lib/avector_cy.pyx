## avector_cy.pyx ##


from . cimport avector as _avector
from . basic_cy cimport *


cdef class AVector:

    def __cinit__(self):
        self.ptr = NULL
        self.owner = False

    def __init__(self, uint dim):
        cdef pavector ptr = _avector.new_avector(dim)
        self._setup(ptr, True)

    def __dealloc(self):
        if self.ptr is not NULL and self.owner is True:
            _avector.del_avector(self.ptr)

    cdef _setup(self, pavector ptr, bint owner):
        self.ptr = ptr
        self.owner = owner
        self.v = <field [:ptr.dim]> (<field *> ptr.v)

    @property
    def dim(self):
        return self.ptr.dim

    @staticmethod
    cdef wrap(pavector ptr, bint owner=False):
        cdef AVector obj = AVector.__new__(AVector)
        obj._setup(ptr, owner)
        return obj