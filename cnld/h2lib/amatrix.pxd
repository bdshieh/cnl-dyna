## amatrix.h ##

from . basic cimport *
from . avector cimport *

cdef extern from 'amatrix.h' nogil:

    struct _amatrix:
        field * a
        uint ld
        uint rows
        uint cols
        
    ctypedef _amatrix amatrix
    ctypedef amatrix * pamatrix
    ctypedef const amatrix * pcamatrix

    pamatrix new_amatrix(uint rows, uint cols)
    pamatrix new_zero_amatrix(uint rows, uint cols)
    # pamatrix new_identity_amatrix(uint rows, uint cols)
    void del_amatrix(pamatrix a)
    # void clear_amatrix(pamatrix a)
    # void identity_amatrix(pamatrix a)
    # void random_amatrix (pamatrix a)
    # void copy_amatrix(bint atrans, pcamatrix a, pamatrix b)
    pamatrix clone_amatrix(pcamatrix src)
    void scale_amatrix(field alpha, pamatrix a)
    void conjugate_amatrix(pamatrix a)
    # real norm2_amatrix(pcamatrix a)
    # real norm2diff_amatrix(pcamatrix a, pcamatrix b)
    void addeval_amatrix_avector(field alpha, pcamatrix a, pcavector src, pavector trg)
    void add_amatrix(field alpha, bint atrans, pcamatrix a, pamatrix b)
    size_t getsize_amatrix(pcamatrix a)
    void addmul_amatrix(field alpha, bint atrans, pcamatrix a, bint btrans, pcamatrix b, pamatrix c)