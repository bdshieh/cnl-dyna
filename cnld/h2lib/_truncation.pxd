## truncation.h ##


from . _basic cimport *


cdef extern from 'truncation.h' nogil:
    
    struct _truncmode:
        bint frobenius
        bint absolute
        bint blocks
        real zeta_level
        real zeta_age

    ctypedef _truncmode truncmode
    ctypedef truncmode * ptruncmode
    ctypedef const truncmode * pctruncmode

    ptruncmode new_truncmode()
    void del_truncmode(ptruncmode tm)
    ptruncmode new_releucl_truncmode()