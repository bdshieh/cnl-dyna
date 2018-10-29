


from . libh2 cimport *


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
    cdef field k
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