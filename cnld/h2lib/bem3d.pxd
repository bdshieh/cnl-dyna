## bem3d_cy.pxd ##


from . cimport _bem3d
from . basic cimport *
from . cluster cimport *
from . block cimport *
from . hmatrix cimport *
from . surface3d cimport *


ctypedef _bem3d.pbem3d pbem3d
ctypedef _bem3d.pcbem3d pcbem3d

cpdef enum basisfunctionbem3d:
    NONE = _bem3d.BASIS_NONE_BEM3D
    CONSTANT = _bem3d.BASIS_CONSTANT_BEM3D
    LINEAR = _bem3d.BASIS_LINEAR_BEM3D

cdef class Bem3d:
    cdef pbem3d ptr
    cdef bint owner
    cdef _setup(self, pbem3d ptr, bint owner)
    @staticmethod
    cdef wrap(pbem3d ptr, bint owner=*)

cpdef build_bem3d_cluster(Bem3d bem, uint clf, basisfunctionbem3d basis)
cpdef setup_hmatrix_aprx_aca_bem3d(Bem3d bem, Cluster rc, Cluster cc, Block tree, real accur)
cpdef setup_hmatrix_aprx_paca_bem3d(Bem3d bem, Cluster rc, Cluster cc, Block tree, real accur)
cpdef setup_hmatrix_aprx_hca_bem3d(Bem3d bem, Cluster rc, Cluster cc, Block tree, uint m, real accur)
cpdef setup_hmatrix_aprx_inter_row_bem3d(Bem3d bem, Cluster rc, Cluster cc, Block tree, uint m)
cpdef assemble_bem3d_hmatrix(Bem3d bem, Block b, HMatrix G)
cpdef assemble_bem3d_amatrix(Bem3d bem, AMatrix G)
