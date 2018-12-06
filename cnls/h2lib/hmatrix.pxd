## hmatrix.h ##

from . basic cimport *
from . amatrix cimport *
from . cluster cimport *
from . block cimport *
from . rkmatrix cimport *
from . sparsematrix cimport *


cdef extern from 'hmatrix.h' nogil:

    # struct _hmatrix

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
        uint refs
        uint desc
        
    phmatrix new_hmatrix(pccluster rc, pccluster cc)
    # phmatrix new_super_hmatrix(pccluster rc, pccluster cc, uint rsons, uint csons)
    void del_hmatrix(phmatrix hm)

    void clear_hmatrix(phmatrix hm)
    void copy_hmatrix(pchmatrix src, phmatrix trg)
    phmatrix clone_hmatrix(pchmatrix src)
    size_t getsize_hmatrix(pchmatrix hm)
    uint getrows_hmatrix(pchmatrix hm)
    uint getcols_hmatrix(pchmatrix hm)
    phmatrix build_from_block_hmatrix(pcblock b, uint k)
    # void norm2_hmatrix(pchmatrix H)
    void addeval_hmatrix_avector(field alpha, pchmatrix hm, pcavector x, pavector y)
    void addevalsymm_hmatrix_avector(field alpha, pchmatrix hm, pcavector x, pavector y)
    phmatrix clonestructure_hmatrix(pchmatrix src)
    void copy_sparsematrix_hmatrix(psparsematrix sp, phmatrix hm)
    