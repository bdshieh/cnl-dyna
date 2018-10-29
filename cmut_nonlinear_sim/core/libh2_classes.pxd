


from . libh2 cimport *


cdef class Vector:
    cdef pavector ptr
    cdef bint owner
    cdef public field [:] v
    @staticmethod
    cdef create(uint dim)
    @staticmethod
    cdef wrap(pavector ptr, bint owner=*)

cdef class Matrix:
    cdef pamatrix ptr
    cdef bint owner
    cdef public field [:,:] a
    @staticmethod
    cdef create(uint rows, uint cols)
    @staticmethod
    cdef wrap(pamatrix ptr, bint owner=*)

cdef class Macrosurface:
    cdef pmacrosurface3d ptr
    cdef bint owner
    cdef public real [:,:] x
    cdef public uint [:,:] e
    cdef public uint [:,:] t
    cdef public uint [:,:] s
    @staticmethod
    cdef create(uint vertices, uint edges, uint triangles)
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
    @staticmethod
    cdef create(uint vertices, uint edges, uint triangles)
    @staticmethod
    cdef wrap(psurface3d ptr, bint owner=*)

cdef class Bem:
    cdef pbem3d ptr
    cdef bint owner
    @staticmethod
    cdef create(pcsurface3d gr, basisfunctionbem3d row_basis, basisfunctionbem3d col_basis)
    @staticmethod
    cdef wrap(pbem3d ptr, bint owner=*)

cdef class Cluster:
    cdef pcluster ptr
    cdef bint owner
    cdef readonly uint [:] idx
    cdef readonly real [:] bmin
    cdef readonly real [:] bmax
    @staticmethod
    cdef create(uint size, uint * idx, uint sons, uint dim)
    @staticmethod
    cdef wrap(pcluster ptr, bint owner=*)

cdef class Block:
    cdef pblock ptr
    cdef bint owner
    @staticmethod
    cdef create(pcluster rc, pcluster cc, bint a, uint rsons, uint csons)
    @staticmethod
    cdef wrap(pblock ptr, bint owner=*)

cdef class RKMatrix:
    cdef prkmatrix ptr
    cdef bint owner
    @staticmethod
    cdef create(uint rows, uint cols, uint k)
    @staticmethod
    cdef wrap(prkmatrix ptr, bint owner=*)

cdef class HMatrix:
    cdef phmatrix ptr
    cdef bint owner
    @staticmethod
    cdef create(pcluster rc, pcluster cc)
    @staticmethod
    cdef wrap(phmatrix ptr, bint owner=*)