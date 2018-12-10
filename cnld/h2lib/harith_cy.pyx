## harith_cy.pxd ##


from . cimport harith as _harith
from . basic_cy cimport *
from . hmatrix_cy  cimport *
from . avector_cy  cimport *
from . truncation_cy  cimport *


cpdef lrdecomp_hmatrix(HMatrix a, Truncmode tm, real eps):
    _harith.lrdecomp_hmatrix(a.ptr, <pctruncmode> tm.ptr, eps)

cpdef lrsolve_hmatrix_avector(bint atrans, HMatrix a, AVector x):
    _harith.lrsolve_hmatrix_avector(atrans, <pchmatrix> a.ptr, x.ptr)

cpdef lreval_hmatrix_avector(bint atrans, HMatrix a, AVector x):
    _harith.lreval_hmatrix_avector(atrans, <pchmatrix> a.ptr, x.ptr)

cpdef choldecomp_hmatrix(HMatrix a, Truncmode tm, real eps):
    _harith.choldecomp_hmatrix(a.ptr, <pctruncmode> tm.ptr, eps)

cpdef cholsolve_hmatrix_avector(HMatrix a, AVector x):
    _harith.cholsolve_hmatrix_avector(<pchmatrix> a.ptr, x.ptr)

cpdef choleval_hmatrix_avector(HMatrix a, AVector x):
    _harith.choleval_hmatrix_avector(<pchmatrix> a.ptr, x.ptr)

cpdef add_hmatrix(field alpha, HMatrix a, Truncmode tm, real eps, HMatrix  b):
    _harith.add_hmatrix(alpha, <pchmatrix> a.ptr, <pctruncmode> tm.ptr, eps, b.ptr)

cpdef addmul_hmatrix(field alpha, bint xtrans, HMatrix x, bint ytrans, HMatrix y, Truncmode tm, real eps, HMatrix z):
    _harith.addmul_hmatrix(alpha, xtrans, <pchmatrix> x.ptr, ytrans, <pchmatrix> y.ptr, <pctruncmode> tm.ptr, eps, z.ptr)