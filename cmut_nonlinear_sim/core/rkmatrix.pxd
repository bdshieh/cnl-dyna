## rkmatrix.h ##

from . basic cimport *
from . amatrix cimport *


cdef extern from 'rkmatrix.h' nogil:

    struct _rkmatrix:
        amatrix A
        amatrix B
        uint k
    
    ctypedef _rkmatrix rkmatrix
    ctypedef rkmatrix * prkmatrix
    ctypedef const rkmatrix * pcrkmatrix

    prkmatrix new_rkmatrix(uint rows, uint cols, uint k)
    void del_rkmatrix(prkmatrix r)