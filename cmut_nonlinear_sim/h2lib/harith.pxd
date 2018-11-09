## harith.h ##


from . basic cimport *
from . hmatrix cimport *
from . avector cimport *
from . truncation cimport *


cdef extern from 'harith.h' nogil:

    void lrdecomp_hmatrix(phmatrix a, pctruncmode tm, real eps)
    void lrsolve_hmatrix_avector(bint atrans, pchmatrix a, pavector x)
    void lreval_hmatrix_avector(bint atrans, pchmatrix a, pavector x)
    void choldecomp_hmatrix(phmatrix a, pctruncmode tm, real eps)
    void cholsolve_hmatrix_avector(pchmatrix a, pavector x)
    void choleval_hmatrix_avector(pchmatrix a, pavector x)