## bem3d.h ##


from . basic cimport *
from . cluster cimport *
from . block cimport *
from . hmatrix cimport *
from . surface3d cimport *


cdef extern from 'bem3d.h' nogil:

    cdef enum _basisfunctionbem3d:
        BASIS_NONE_BEM3D
        BASIS_CONSTANT_BEM3D
        BASIS_LINEAR_BEM3D
    
    ctypedef _basisfunctionbem3d basisfunctionbem3d

    struct _bem3d:
        field k
        field kernel_const

    ctypedef _bem3d bem3d
    ctypedef bem3d * pbem3d
    ctypedef const bem3d * pcbem3d

    pbem3d new_bem3d(pcsurface3d gr, basisfunctionbem3d row_basis, basisfunctionbem3d col_basis)
    void del_bem3d(pbem3d bem)

    pcluster build_bem3d_cluster(pcbem3d bem, uint clf, basisfunctionbem3d basis)
    void setup_hmatrix_aprx_aca_bem3d(pbem3d bem, pccluster rc, pccluster cc, pcblock tree, real accur)
    void setup_hmatrix_aprx_paca_bem3d(pbem3d bem, pccluster rc, pccluster cc, pcblock tree, real accur)
    void setup_hmatrix_aprx_hca_bem3d(pbem3d bem, pccluster rc, pccluster cc, pcblock tree, uint m, real accur)
    void setup_hmatrix_aprx_inter_row_bem3d(pbem3d bem, pccluster rc, pccluster cc, pcblock tree, uint m)
    void assemble_bem3d_hmatrix(pbem3d bem, pblock b, phmatrix G)
