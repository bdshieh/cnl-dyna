## harith_cy.pxd ##


from . cimport harith as _harith
from . basic_cy cimport *
from . hmatrix_cy  cimport *
from . avector_cy  cimport *
from . truncation_cy  cimport *


cpdef lrdecomp_hmatrix(HMatrix a, Truncmode tm, real eps)
cpdef lrsolve_hmatrix_avector(bint atrans, HMatrix a, AVector x)
cpdef lreval_hmatrix_avector(bint atrans, HMatrix a, AVector x)
cpdef choldecomp_hmatrix(HMatrix a, Truncmode tm, real eps)
cpdef cholsolve_hmatrix_avector(HMatrix a, AVector x)
cpdef choleval_hmatrix_avector(HMatrix a, AVector x)
cpdef add_hmatrix(field alpha, HMatrix a, Truncmode tm, real eps, HMatrix  b)
cpdef addmul_hmatrix(field alpha, bint xtrans, HMatrix x, bint ytrans, HMatrix y, Truncmode tm, real eps, HMatrix z)