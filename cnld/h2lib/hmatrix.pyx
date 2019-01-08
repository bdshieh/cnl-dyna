## hmatrix_cy.pyx ##


from . cimport _hmatrix
from . basic cimport *
from . amatrix cimport *
from . cluster cimport *
from . block cimport *
from . rkmatrix cimport *
from . sparsematrix cimport *


cdef class HMatrix:

    ''' Initialization methods '''
    def __cinit__(self):
        self.ptr = NULL
        self.owner = False

    def __init__(self, Cluster rc, Cluster cc):
        cdef phmatrix ptr = _hmatrix.new_hmatrix(rc.ptr, cc.ptr)
        self._setup(ptr, owner=True)

    def __dealloc__(self):
        if self.ptr is not NULL and self.owner is True:
            _hmatrix.del_hmatrix(self.ptr)

    cdef _setup(self, phmatrix ptr, bint owner):
        self.ptr = ptr
        self.owner = owner

    ''' Scalar properties '''
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
    def r(self):
        if self.ptr.r is not NULL:
            return RKMatrix.wrap(self.ptr.r, False)

    @property
    def f(self):
        if self.ptr.f is not NULL:
            return AMatrix.wrap(self.ptr.f, False)

    @property
    def son(self):
        return [HMatrix.wrap(self.ptr.son[i], False) for i in range(self.rsons + self.csons)]

    ''' Methods '''
    @staticmethod
    cdef wrap(phmatrix ptr, bint owner=False):
        cdef HMatrix obj = HMatrix.__new__(HMatrix)
        obj._setup(ptr, owner)
        return obj
    
cpdef HMatrix build_from_block_hmatrix(Block b, uint k):

    cdef phmatrix Z = _hmatrix.build_from_block_hmatrix(<pcblock> b.ptr, k)
    return HMatrix.wrap(Z, True)

cpdef clear_hmatrix(HMatrix hm):
    _hmatrix.clear_hmatrix(hm.ptr)

cpdef copy_hmatrix(HMatrix src, HMatrix trg):
    _hmatrix.copy_hmatrix(<pchmatrix> src.ptr, trg.ptr)

cpdef HMatrix clone_hmatrix(HMatrix src):
    cdef phmatrix cpy = _hmatrix.clone_hmatrix(<pchmatrix> src.ptr)
    return HMatrix.wrap(cpy, True)

cpdef size_t getsize_hmatrix(HMatrix hm):
    return _hmatrix.getsize_hmatrix(<pchmatrix> hm.ptr)

cpdef addeval_hmatrix_avector(field alpha, HMatrix hm, AVector x, AVector y):
    _hmatrix.addeval_hmatrix_avector(alpha, <pchmatrix> hm.ptr, <pcavector> x.ptr, y.ptr)

cpdef addevalsymm_hmatrix_avector(field alpha, HMatrix hm, AVector x, AVector y):
    _hmatrix.addevalsymm_hmatrix_avector(alpha, <pchmatrix> hm.ptr, <pcavector> x.ptr, y.ptr)

cpdef uint getrows_hmatrix(HMatrix hm):
    return _hmatrix.getrows_hmatrix(<pchmatrix> hm.ptr)
    
cpdef uint getcols_hmatrix(HMatrix hm):
    return _hmatrix.getcols_hmatrix(<pchmatrix> hm.ptr)

cpdef copy_sparsematrix_hmatrix(SparseMatrix sp, HMatrix hm):
    _hmatrix.copy_sparsematrix_hmatrix(sp.ptr, hm.ptr)

cpdef HMatrix clonestructure_hmatrix(HMatrix src):
    cpdef phmatrix hm = _hmatrix.clonestructure_hmatrix(<pchmatrix> src.ptr)
    return HMatrix.wrap(hm, True)

cpdef identity_hmatrix(HMatrix hm):
    _hmatrix.identity_hmatrix(hm.ptr)