## vector.h ##

from . basic cimport *

cdef extern from 'avector.h' nogil:
    
    struct _avector:
        field * v
        uint dim

    ctypedef _avector avector
    ctypedef avector * pavector
    ctypedef const avector * pcavector

    pavector new_avector(uint dim)
    pavector new_zero_avector(uint dim)
    void del_avector(pavector v)
    void add_avector(field alpha, pcavector x, pavector y)
    void clear_avector(pavector v)
    void fill_avector(pavector v, field x)
    void scale_avector(field alpha, pavector v)
    void copy_avector(pavector v, pavector w)
    void random_avector(pavector v)
    real norm2_avector(pcavector v)