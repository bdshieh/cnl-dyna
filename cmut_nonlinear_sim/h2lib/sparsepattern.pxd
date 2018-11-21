## sparsepattern.h ##

from . basic cimport *

cdef extern from 'sparsematrix.h' nogil:

    ctypedef _sparsepattern sparsepattern
    ctypedef sparsepattern * psparsepattern
    ctypedef const sparsepattern * pcsparsepattern

    ctypedef _patentry patentry
    ctypedef patentry * ppatentry
    ctypedef const patentry * pcpatentry

    struct _sparsepattern:
        uint rows
        uint cols
        ppatentry * row
    
    struct _patentry
        uint row
        uint col
        struct _patentry * next

    psparsepattern new_sparsepattern(uint rows, uint cols)
    void del_sparsepattern(psparsepattern sp)
    void clear_sparsepattern(psparsepattern sp)
    void addnz_sparsepattern(psparsepattern sp, uint row, uint col)
    
    