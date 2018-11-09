## block_cy.pyx ##


from . cimport block as _block
from . basic_cy cimport *
from . cluster_cy cimport *


cdef class Block:

    def __cinit__(self):
        self.ptr = NULL
        self.owner = False

    def __init__(self, Cluster rc, Cluster cc, bint a, uint rsons, uint csons):
        cdef pblock ptr = _block.new_block(rc.ptr, cc.ptr, a, rsons, csons)
        self._setup(ptr, True)

    def __dealloc(self):
        if self.ptr is not NULL and self.owner is True:
            _block.del_block(self.ptr)

    cdef _setup(self, pblock ptr, bint owner):
        self.ptr = ptr
        self.owner = owner

    @property
    def a(self):
        return self.ptr.a

    @property
    def rsons(self):
        return self.ptr.rsons

    @property
    def csons(self):
        return self.ptr.csons

    @property
    def desc(self):
        return self.ptr.desc

    @staticmethod
    cdef wrap(pblock ptr, bint owner=False):
        cdef Block obj = Block.__new__(Block)
        obj._setup(ptr, owner)
        return obj

cpdef build_nonstrict_block(Cluster rc, Cluster cc, real eta, str admis):
    
    cdef admissible _admis

    if admis.lower() in ['2']:
        _admis = admissible_2_cluster
    elif admis.lower() in ['max']:
        _admis = admissible_max_cluster
    elif admis.lower() in ['sphere']:
        _admis = admissible_sphere_cluster
    elif admis.lower() in ['2min', '2_min']:
        _admis = admissible_2_min_cluster
    else:
        raise TypeError

    cdef pblock block = _block.build_nonstrict_block(rc.ptr, cc.ptr, &eta, _admis)
    return Block.wrap(block, True)