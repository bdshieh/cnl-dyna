## factorizations_cy.pxd ##


from . cimport _factorizations
from . basic cimport *
from . amatrix  cimport *
from . avector  cimport *
from . truncation  cimport *


cpdef uint lrdecomp_amatrix(AMatrix a)
cpdef lrsolve_amatrix_avector(bint atrans, AMatrix a, AVector x)
cpdef choldecomp_amatrix(AMatrix a)
cpdef cholsolve_amatrix_avector(AMatrix a, AVector x)

