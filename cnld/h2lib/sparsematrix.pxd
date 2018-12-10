## sparsematrix.h ##

from . basic cimport *
from . amatrix cimport *

cdef extern from 'sparsematrix.h' nogil:

    ctypedef field * pfield
    ctypedef _sparsematrix sparsematrix
    ctypedef sparsematrix * psparsematrix
    ctypedef const sparsematrix * pcsparsematrix

    struct _sparsematrix:
        uint rows
        uint cols
        uint nz
        uint * row
        uint * col
        pfield coeff
    
    psparsematrix new_raw_sparsematrix(uint rows, uint cols, uint nz)
    psparsematrix new_identity_sparsematrix(uint rows, uint cols)
    void del_sparsematrix(psparsematrix a)
    field addentry_sparsematrix(psparsematrix a, uint row, uint col, field x)
    void setentry_sparsematrix(psparsematrix a, uint row, uint col, field x)
    size_t getsize_sparsematrix(pcsparsematrix a)
    void sort_sparsematrix(psparsematrix a)
    void clear_sparsematrix(psparsematrix a)
    void add_sparsematrix_amatrix(field alpha, bint atrans, pcsparsematrix a, pamatrix b)
    