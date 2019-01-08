## factorizations_cy.pyx ##


from . cimport _factorizations
from . basic cimport *
from . amatrix  cimport *
from . avector  cimport *
from . truncation  cimport *


cpdef uint lrdecomp_amatrix(AMatrix a):
    return _factorizations.lrdecomp_amatrix(a.ptr)

cpdef lrsolve_amatrix_avector(bint atrans, AMatrix a, AVector x):
    _factorizations.lrsolve_amatrix_avector(atrans, <pcamatrix> a.ptr, x.ptr)

cpdef choldecomp_amatrix(AMatrix a):
    _factorizations.choldecomp_amatrix(a.ptr)

cpdef cholsolve_amatrix_avector(AMatrix a, AVector x):
    _factorizations.cholsolve_amatrix_avector(<pcamatrix> a.ptr, x.ptr)


