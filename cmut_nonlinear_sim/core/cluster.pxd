## cluster.h ##

from . basic cimport *

cdef extern from 'cluster.h' nogil:
    
    struct _cluster:
        uint size
        uint * idx
        uint sons
        uint dim
        real * bmin
        real * bmax
        uint desc
        uint type

    ctypedef _cluster cluster
    ctypedef cluster * pcluster
    ctypedef const cluster * pccluster

    pcluster new_cluster(uint size, uint * idx, uint sons, uint dim)
    void del_cluster(pcluster t)