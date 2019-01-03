## harith.h ##


from . _basic cimport *
from . _hmatrix cimport *
from . _avector cimport *
from . _amatrix cimport *
from . _truncation cimport *


cdef extern from 'harith.h' nogil:

    void lrdecomp_hmatrix(phmatrix a, pctruncmode tm, real eps)
    void lrsolve_hmatrix_avector(bint atrans, pchmatrix a, pavector x)
    void lreval_hmatrix_avector(bint atrans, pchmatrix a, pavector x)
    void choldecomp_hmatrix(phmatrix a, pctruncmode tm, real eps)
    void cholsolve_hmatrix_avector(pchmatrix a, pavector x)
    void choleval_hmatrix_avector(pchmatrix a, pavector x)
    void addmul_hmatrix(field alpha, bint xtrans, pchmatrix x, bint ytrans, pchmatrix y, pctruncmode tm, real eps, phmatrix z)
    void add_hmatrix(field alpha, pchmatrix a, pctruncmode tm, real eps, phmatrix  b)
    void add_hmatrix_amatrix(field alpha, bint atrans, pchmatrix a, pamatrix b)
    void add_amatrix_hmatrix(field alpha, bint atrans, pcamatrix a, pctruncmode tm, real eps, phmatrix b)
