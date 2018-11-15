## block_cy.pxd ##


from . cimport block as _block
from . basic_cy cimport *
from . cluster_cy cimport *


ctypedef _block.pblock pblock
ctypedef _block.pcblock pcblock
ctypedef _block.admissible admissible

cdef class Block:
    cdef pblock ptr
    cdef bint owner
    cdef _setup(self, pblock ptr, bint owner)
    @staticmethod
    cdef wrap(pblock ptr, bint owner=*)

from . block cimport admissible_2_cluster, admissible_max_cluster, admissible_sphere_cluster, admissible_2_min_cluster
cpdef build_nonstrict_block(Cluster rc, Cluster cc, real eta, str admis)
cpdef build_strict_block(Cluster rc, Cluster cc, real eta, str admis)