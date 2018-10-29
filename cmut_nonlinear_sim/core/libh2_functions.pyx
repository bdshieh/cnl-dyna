

from . libh2 cimport *
from . libh2_classes cimport *


def init_libh2():
    init_h2lib(NULL, NULL)

def uninit_libh2():
    uninit_h2lib()

cpdef build_from_macrosurface_surface(Macrosurface ms, uint refn):

    cdef psurface3d surf = build_from_macrosurface3d_surface3d(<pcmacrosurface3d> ms.ptr, refn)
    return Surface.wrap(surf, True)



# cpdef new_slp_helmholtz_bem3d(field k, Surface3D gr, uint q_reg, uint q_sing, basisfunctionbem3d basis,  
#         basisfunctionbem3d basis):
#     pass


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