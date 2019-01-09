## harith_cy.pxd ##


from . cimport _harith
from . basic cimport *
from . hmatrix  cimport *
from . amatrix cimport *
from . avector  cimport *
from . truncation  cimport *


cpdef lrdecomp_hmatrix(HMatrix a, Truncmode tm, real eps)
cpdef lrsolve_hmatrix_avector(bint atrans, HMatrix a, AVector x)
cpdef lreval_hmatrix_avector(bint atrans, HMatrix a, AVector x)
cpdef choldecomp_hmatrix(HMatrix a, Truncmode tm, real eps)
cpdef cholsolve_hmatrix_avector(HMatrix a, AVector x)
cpdef choleval_hmatrix_avector(HMatrix a, AVector x)
cpdef add_hmatrix(field alpha, HMatrix a, Truncmode tm, real eps, HMatrix  b)
cpdef add_hmatrix_amatrix(field alpha, bint atrans, HMatrix a, AMatrix b)
cpdef addmul_hmatrix(field alpha, bint xtrans, HMatrix x, bint ytrans, HMatrix y, Truncmode tm, real eps, HMatrix z)
cpdef add_amatrix_hmatrix(field alpha, bint atrans, AMatrix a, Truncmode tm, real eps, HMatrix b)
cpdef triangularsolve_hmatrix_avector(bint alower, bint aunit, bint atrans, HMatrix a, AVector x)