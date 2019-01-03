## block.h ##

from . _basic cimport *
from . _cluster cimport *



cdef extern from 'block.h' nogil:

    ctypedef _block block
    ctypedef block * pblock
    ctypedef const block * pcblock
    ctypedef bint (* admissible) (pcluster rc, pcluster cc, void * data)

    struct _block:
        pcluster rc
        pcluster cc
        bint a
        pblock * son
        uint rsons
        uint csons
        uint desc

    pblock new_block(pcluster rc, pcluster cc, bint a, uint rsons, uint csons)
    void del_block(pblock b)
    pblock build_nonstrict_block(pcluster rc, pcluster cc, void * data, admissible admis)
    pblock build_strict_block(pcluster rc, pcluster cc, void * data, admissible admis)
    bint admissible_2_cluster(pcluster rc, pcluster cc, void * data)
    bint admissible_max_cluster(pcluster rc, pcluster cc, void * data)
    bint admissible_sphere_cluster(pcluster rc, pcluster cc, void * data)
    bint admissible_2_min_cluster(pcluster rc, pcluster cc, void * data)