## factorizations.h ##


# from . blas cimport *
from . _amatrix cimport *
from . _avector cimport *
# from . truncation cimport *


cdef extern from 'factorizations.h' nogil:

    void lrdecomp_amatrix(pamatrix a)
    void lrsolve_amatrix_avector(bint atrans, pcamatrix a, pavector x)
    # void lreval_hmatrix_avector(bint atrans, pchmatrix a, pavector x)
    void choldecomp_amatrix(pamatrix a)
    void cholsolve_amatrix_avector(pcamatrix a, pavector x)
    # void choleval_hmatrix_avector(pchmatrix a, pavector x)