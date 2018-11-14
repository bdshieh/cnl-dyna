## factorizations_cy.pyx ##


from . cimport factorizations as _factorizations
from . basic_cy cimport *
from . amatrix_cy  cimport *
from . avector_cy  cimport *
from . truncation_cy  cimport *


cpdef lrdecomp_amatrix(AMatrix a):
    _factorizations.lrdecomp_amatrix(a.ptr)

cpdef lrsolve_amatrix_avector(bint atrans, AMatrix a, AVector x):
    _factorizations.lrsolve_amatrix_avector(atrans, <pcamatrix> a.ptr, x.ptr)

cpdef choldecomp_amatrix(AMatrix a):
    _factorizations.choldecomp_amatrix(a.ptr)

cpdef cholsolve_amatrix_avector(AMatrix a, AVector x):
    _factorizations.cholsolve_amatrix_avector(<pcamatrix> a.ptr, x.ptr)


