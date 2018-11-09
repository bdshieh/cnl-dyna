## bem3d_cy.pxd ##


from . cimport bem3d as _bem3d
from . basic_cy cimport *
from . cluster_cy cimport *
from . block_cy cimport *
from . hmatrix_cy cimport *
from . surface3d_cy cimport *


cdef class Bem3d:

    def __cinit__(self):
        self.ptr = NULL
        self.owner = False

    def __init__(self, Surface3d surf, basisfunctionbem3d row_basis, basisfunctionbem3d col_basis):
        cdef pbem3d ptr = _bem3d.new_bem3d(<pcsurface3d> surf.ptr, row_basis, col_basis)
        self._setup(ptr, True)

    def __dealloc(self):
        if self.ptr is not NULL and self.owner is True:
            _bem3d.del_bem3d(self.ptr)

    cdef _setup(self, pbem3d ptr, bint owner):
        self.ptr = ptr
        self.owner = owner

    @property
    def k(self):
        return self.ptr.k

    @property
    def kernel_const(self):
        return self.ptr.kernel_const
        
    @staticmethod
    cdef wrap(pbem3d ptr, bint owner=False):
        cdef Bem3d obj = Bem3d.__new__(Bem3d)
        obj._setup(ptr, owner)
        return obj

cpdef build_bem3d_cluster(Bem3d bem, uint clf, basisfunctionbem3d basis):

    cdef pcluster cluster = _bem3d.build_bem3d_cluster(<pcbem3d> bem.ptr, clf, basis)
    return Cluster.wrap(cluster, True)

cpdef setup_hmatrix_aprx_aca_bem3d(Bem3d bem, Cluster rc, Cluster cc, Block tree, real accur):
    _bem3d.setup_hmatrix_aprx_aca_bem3d(<pbem3d> bem.ptr, <pccluster> rc.ptr, <pccluster> cc.ptr, <pcblock> tree, accur)

cpdef setup_hmatrix_aprx_paca_bem3d(Bem3d bem, Cluster rc, Cluster cc, Block tree, real accur):
    _bem3d.setup_hmatrix_aprx_paca_bem3d(<pbem3d> bem.ptr, <pccluster> rc.ptr, <pccluster> cc.ptr, <pcblock> tree, accur)

cpdef setup_hmatrix_aprx_hca_bem3d(Bem3d bem, Cluster rc, Cluster cc, Block tree, uint m, real accur):
    _bem3d.setup_hmatrix_aprx_hca_bem3d(<pbem3d> bem.ptr, <pccluster> rc.ptr, <pccluster> cc.ptr, <pcblock> tree, m, accur)

cpdef setup_hmatrix_aprx_inter_row_bem3d(Bem3d bem, Cluster rc, Cluster cc, Block tree, uint m):
    _bem3d.setup_hmatrix_aprx_inter_row_bem3d(<pbem3d> bem.ptr, <pccluster> rc.ptr, <pccluster> cc.ptr, <pcblock> tree, m)