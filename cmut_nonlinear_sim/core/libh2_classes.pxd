## libh2_classes.pxd ##

# from . basic cimport *
# from . avector cimport *
# from . amatrix cimport *
# from . cluster cimport *
# from . block cimport *
# from . surface3d cimport *
# from . macrosurface3d cimport *
# from . bem3d cimport *
# from . rkmatrix cimport *
# from . hmatrix cimport *
from . cimport libh2

ctypedef libh2.real real
ctypedef libh2.field field
ctypedef libh2.uint uint
ctypedef libh2.pavector pavector
ctypedef libh2.pamatrix pamatrix
ctypedef libh2.pmacrosurface3d pmacrosurface3d
ctypedef libh2.pcmacrosurface3d pcmacrosurface3d
ctypedef libh2.psurface3d psurface3d
ctypedef libh2.pcsurface3d pcsurface3d
ctypedef libh2.pbem3d pbem3d
ctypedef libh2.pcbem3d pcbem3d
ctypedef libh2.phmatrix phmatrix
ctypedef libh2.prkmatrix prkmatrix
ctypedef libh2.pcluster pcluster
ctypedef libh2.pccluster pccluster
ctypedef libh2.pblock pblock
ctypedef libh2.pcblock pcblock
ctypedef libh2.basisfunctionbem3d basisfunctionbem3d
ctypedef libh2.admissible admissible


cpdef enum basistype:
    NONE = libh2.BASIS_NONE_BEM3D
    CONSTANT = libh2.BASIS_CONSTANT_BEM3D
    LINEAR = libh2.BASIS_LINEAR_BEM3D

cdef class Vector:
    cdef pavector ptr
    cdef bint owner
    cdef public field [:] v
    cdef _setup(self, pavector ptr, bint owner)
    @staticmethod
    cdef wrap(pavector ptr, bint owner=*)

cdef class Matrix:
    cdef pamatrix ptr
    cdef bint owner
    cdef public field [:,:] a
    cdef _setup(self, pamatrix ptr, bint owner)
    @staticmethod
    cdef wrap(pamatrix ptr, bint owner=*)

cdef class Macrosurface:
    cdef pmacrosurface3d ptr
    cdef bint owner
    cdef public real [:,:] x
    cdef public uint [:,:] e
    cdef public uint [:,:] t
    cdef public uint [:,:] s
    cdef _setup(self, pmacrosurface3d ptr, bint owner)
    @staticmethod
    cdef wrap(pmacrosurface3d ptr, bint owner=*)

cdef class Surface:
    cdef psurface3d ptr
    cdef bint owner
    cdef public real [:,:] x
    cdef public uint [:,:] e
    cdef public uint [:,:] t
    cdef public uint [:,:] s
    cdef public real [:,:] n
    cdef _setup(self, psurface3d ptr, bint owner)
    @staticmethod
    cdef wrap(psurface3d ptr, bint owner=*)

cdef class Bem:
    cdef pbem3d ptr
    cdef bint owner
    cdef _setup(self, pbem3d ptr, bint owner)
    @staticmethod
    cdef wrap(pbem3d ptr, bint owner=*)

cdef class Cluster:
    cdef pcluster ptr
    cdef bint owner
    cdef readonly uint [:] idx
    cdef readonly real [:] bmin
    cdef readonly real [:] bmax
    cdef _setup(self, pcluster ptr, bint owner)
    @staticmethod
    cdef wrap(pcluster ptr, bint owner=*)

cdef class Block:
    cdef pblock ptr
    cdef bint owner
    cdef _setup(self, pblock ptr, bint owner)
    @staticmethod
    cdef wrap(pblock ptr, bint owner=*)

cdef class RKMatrix:
    cdef prkmatrix ptr
    cdef bint owner
    cdef _setup(self, prkmatrix ptr, bint owner)
    @staticmethod
    cdef wrap(prkmatrix ptr, bint owner=*)

cdef class HMatrix:
    cdef phmatrix ptr
    cdef bint owner
    cdef _setup(self, phmatrix ptr, bint owner)
    @staticmethod
    cdef wrap(phmatrix ptr, bint owner=*)