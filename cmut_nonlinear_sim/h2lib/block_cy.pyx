## block_cy.pyx ##


from . cimport block as _block
from . basic_cy cimport *
from . cluster_cy cimport *


cdef class Block:

    ''' Initialization methods '''
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

    ''' Scalar properties '''
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

    ''' Pointer properties '''
    @property
    def rc(self):
        return Cluster.wrap(self.ptr.rc, False)

    @property
    def cc(self):
        return Cluster.wrap(self.ptr.cc, False)

    @property
    def son(self):
        return [Block.wrap(self.ptr.son[i], False) for i in range(self.rsons + self.csons)]
    
    ''' Methods '''
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


cpdef build_strict_block(Cluster rc, Cluster cc, real eta, str admis):
    
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

    cdef pblock block = _block.build_strict_block(rc.ptr, cc.ptr, &eta, _admis)
    return Block.wrap(block, True)