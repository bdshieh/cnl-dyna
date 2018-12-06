## cluster_cy.pxd ##


from . cimport cluster as _cluster
from . basic_cy cimport *


ctypedef _cluster.pcluster pcluster
ctypedef _cluster.pccluster pccluster

cdef class Cluster:
    cdef pcluster ptr
    cdef bint owner
    cdef readonly uint [:] _idx
    cdef readonly real [:] _bmin
    cdef readonly real [:] _bmax
    cdef _setup(self, pcluster ptr, bint owner)
    @staticmethod
    cdef wrap(pcluster ptr, bint owner=*)