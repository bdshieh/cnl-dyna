## hmatrix.h ##

from . basic cimport *
from . amatrix cimport *
from . cluster cimport *
from . block cimport *
from . rkmatrix cimport *


cdef extern from 'hmatrix.h' nogil:

    struct _hmatrix

    ctypedef _hmatrix hmatrix
    ctypedef hmatrix * phmatrix
    ctypedef const hmatrix * pchmatrix

    struct _hmatrix:
        pccluster rc
        pccluster cc
        prkmatrix r
        pamatrix f
        phmatrix * son
        uint rsons
        uint csons
        uint desc
        
    phmatrix new_hmatrix(pccluster rc, pccluster cc)
    phmatrix new_super_hmatrix(pccluster rc, pccluster cc, uint rsons, uint csons)
    void del_hmatrix(phmatrix hm)

    void clear_hmatrix(hmatrix hm)
    void copy_hmatrix(pchmatrix src, phmatrix trg)
    phmatrix clone_hmatrix(hmatrix hm)
    size_t getsize_hmatrix(hmatrix hm)
    void build_from_block_hmatrix(pcblock b, uint k)
    void norm2_hmatrix(pchmatrix H)