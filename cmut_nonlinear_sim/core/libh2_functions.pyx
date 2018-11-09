## libh2_functions.pyx ##

from . libh2_classes cimport *
# from . libh2_classes import *


def init_h2lib():
    libh2.init_h2lib(NULL, NULL)


def uninit_h2lib():
    libh2.uninit_h2lib()


cpdef build_from_macrosurface3d_surface3d(Macrosurface ms, uint refn):

    cdef psurface3d surf = libh2.build_from_macrosurface3d_surface3d(<pcmacrosurface3d> ms.ptr, refn)
    return Surface.wrap(surf, True)


cpdef new_slp_helmholtz_bem3d(field k, Surface surf, uint q_regular, uint q_singular,
    basistype row_basis, basistype col_basis):

    cdef pbem3d bem = libh2.new_slp_helmholtz_bem3d(k, <pcsurface3d> surf.ptr, q_regular, q_singular, 
        <basisfunctionbem3d> row_basis, <basisfunctionbem3d> col_basis)
    return Bem.wrap(bem, True)


cpdef new_dlp_helmholtz_bem3d(field k, Surface surf, uint q_regular, uint q_singular, 
    basistype row_basis, basistype col_basis, field alpha):

    cdef pbem3d bem = libh2.new_dlp_helmholtz_bem3d(k, <pcsurface3d> surf.ptr, q_regular, q_singular, 
        <basisfunctionbem3d> row_basis, <basisfunctionbem3d> col_basis, alpha)
    return Bem.wrap(bem, True)


cpdef build_bem3d_cluster(Bem bem, uint clf, basistype basis):

    cdef pcluster cluster = libh2.build_bem3d_cluster(<pcbem3d> bem.ptr, clf, <basisfunctionbem3d> basis)
    return Cluster.wrap(cluster, True)


cpdef build_nonstrict_block(Cluster rc, Cluster cc, real eta, str admissibility):
    
    cdef admissible admis

    if admissibility.lower() in ['2']:
        admis = libh2.admissible_2_cluster
    elif admissibility.lower() in ['maxcluster']:
        admis = libh2.admissible_max_cluster

    cdef pblock block = libh2.build_nonstrict_block(rc.ptr, cc.ptr, &eta, admis)
    return Block.wrap(block, True)


cpdef setup_hmatrix_aprx_aca_bem3d(Bem bem, Cluster rc, Cluster cc, Block tree, real accur):
    libh2.setup_hmatrix_aprx_aca_bem3d(<pbem3d> bem.ptr, <pccluster> rc.ptr, <pccluster> cc.ptr, <pcblock> tree, accur)

cpdef setup_hmatrix_aprx_paca_bem3d(Bem bem, Cluster rc, Cluster cc, Block tree, real accur):
    libh2.setup_hmatrix_aprx_paca_bem3d(<pbem3d> bem.ptr, <pccluster> rc.ptr, <pccluster> cc.ptr, <pcblock> tree, accur)

cpdef setup_hmatrix_aprx_hca_bem3d(Bem bem, Cluster rc, Cluster cc, Block tree, uint m, real accur):
    libh2.setup_hmatrix_aprx_hca_bem3d(<pbem3d> bem.ptr, <pccluster> rc.ptr, <pccluster> cc.ptr, <pcblock> tree, m, accur)

cpdef setup_hmatrix_aprx_inter_row_bem3d(Bem bem, Cluster rc, Cluster cc, Block tree, uint m):
    libh2.setup_hmatrix_aprx_inter_row_bem3d(<pbem3d> bem.ptr, pccluster> rc.ptr, <pccluster> cc.ptr, <pcblock> tree, m)

