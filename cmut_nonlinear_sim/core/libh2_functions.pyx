

from . libh2 cimport *
from . libh2_classes cimport *


def init_libh2():
    init_h2lib(NULL, NULL)

def uninit_libh2():
    uninit_h2lib()

cpdef build_from_macrosurface_surface(Macrosurface ms, uint refn):

    cdef psurface3d surf = build_from_macrosurface3d_surface3d(<pcmacrosurface3d> ms.ptr, refn)
    return Surface.wrap(surf, True)

cpdef new_slp_bem(field k, Surface surf, uint q_regular, uint q_singular, row_basis, col_basis):

    cdef basisfunctionbem3d rb = BASIS_CONSTANT_BEM3D
    cdef basisfunctionbem3d cb = BASIS_CONSTANT_BEM3D

    # if row_basis.lower() in ['const', 'constant']:
    #     rb = BASIS_CONSTANT_BEM3D
    # elif row_basis.lower() in ['lin', 'linear']:
    #     rb = BASIS_LINEAR_BEM3D
    # else:
    #     raise TypeError

    # if col_basis.lower() in ['const', 'constant']:
    #     cb = BASIS_CONSTANT_BEM3D
    # elif col_basis.lower() in ['lin', 'linear']:
    #     cb = BASIS_LINEAR_BEM3D
    # else:
    #     raise TypeError

    cdef pbem3d bem = new_slp_helmholtz_bem3d(k, <pcsurface3d> surf.ptr, q_regular, q_singular, rb, cb)
    print(<unsigned long> bem)
    return Bem.wrap(bem, True)

# cpdef new_dlp_bem(field k, Surface surf, uint q_regular, uint q_singular, row_basis, col_basis, field alpha):

#     cdef basisfunctionbem3d rb = BASIS_CONSTANT_BEM3D
#     cdef basisfunctionbem3d cb = BASIS_CONSTANT_BEM3D
#     cdef pbem3d bem

#     bem = new_dlp_helmholtz_bem3d(k, <pcsurface3d> surf.ptr, q_regular, q_singular, rb, cb, alpha)
#     return Bem.wrap(bem, True)


# cpdef build_bem3d_cluster(Bem3d bem_slp, uint clf, basisfunctionbem3d basis):
#     pass


# cpdef build_nonstrict_block(Block root, Block root, real &eta, admissible admissible_2_cluster):
#     pass


# cdef void cube_parametrization(uint i, real xr1, real xr2, void * phidata, real xt[3]):

#     cdef pcmacrosurface3d mg = <pcmacrosurface3d> data
#     cdef const real (* x)[3]
#     x = <const real ( *)[3]> mg.x
#     cdef const uint (* t)[3]
#     t = <const uint ( *)[3]> mg.t

#     assert(i < mg.triangles)
#     assert(t[i][0] < mg.vertices)
#     assert(t[i][1] < mg.vertices)
#     assert(t[i][2] < mg.vertices)

#     xt[0] = (x[t[i][0]][0] * (1.0 - xr1 - xr2) + x[t[i][1]][0] * xr1 + x[t[i][2]][0] * xr2)
#     xt[1] = (x[t[i][0]][1] * (1.0 - xr1 - xr2) + x[t[i][1]][1] * xr1 + x[t[i][2]][1] * xr2)
#     xt[2] = (x[t[i][0]][2] * (1.0 - xr1 - xr2) + x[t[i][1]][2] * xr1 + x[t[i][2]][2] * xr2)