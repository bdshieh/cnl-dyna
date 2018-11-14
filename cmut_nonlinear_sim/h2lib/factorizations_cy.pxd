## factorizations_cy.pxd ##


from . cimport factorizations as _factorizations
from . basic_cy cimport *
from . amatrix_cy  cimport *
from . avector_cy  cimport *
from . truncation_cy  cimport *


cpdef lrdecomp_amatrix(AMatrix a)
cpdef lrsolve_amatrix_avector(bint atrans, AMatrix a, AVector x)
cpdef choldecomp_amatrix(AMatrix a)
cpdef cholsolve_amatrix_avector(AMatrix a, AVector x)

