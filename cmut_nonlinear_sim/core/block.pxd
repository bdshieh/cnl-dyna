## block.h ##

from . basic cimport *
from . cluster cimport *



cdef extern from 'block.h' nogil:

    struct _block:
        pcluster rc
        pcluster cc
        bint a
        uint rsons
        uint csons
        uint desc
        
    ctypedef _block block
    ctypedef block * pblock
    ctypedef const block * pcblock
    ctypedef bint (* admissible) (pcluster rc, pcluster cc, void * data)

    pblock new_block(pcluster rc, pcluster cc, bint a, uint rsons, uint csons)
    void del_block(pblock b)
    pblock build_nonstrict_block(pcluster rc, pcluster cc, void * data, admissible admis)