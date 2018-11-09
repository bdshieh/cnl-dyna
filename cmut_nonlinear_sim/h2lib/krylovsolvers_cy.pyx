## krylovsolvers_cy.pyx ##


from . cimport krylovsolvers as _krylovsolvers
from . basic_cy cimport *
from . amatrix_cy cimport *
from . hmatrix_cy cimport *


cpdef uint solve_cg_amatrix_avector(AMatrix A, AVector b, AVector x, real eps, uint maxiter):
    return _krylovsolvers.solve_cg_amatrix_avector(<pcamatrix> A.ptr, <pcavector> b.ptr, x.ptr, eps, maxiter)

cpdef uint solve_cg_hmatrix_avector(HMatrix A, AVector b, AVector x, real eps, uint maxiter):
    return _krylovsolvers.solve_cg_hmatrix_avector(<pchmatrix> A.ptr, <pcavector> b.ptr, x.ptr, eps, maxiter)

cpdef uint solve_gmres_amatrix_avector(AMatrix A, AVector b, AVector x, real eps, uint maxiter, uint kmax):
    return _krylovsolvers.solve_gmres_amatrix_avector(<pcamatrix> A.ptr, <pcavector> b.ptr, x.ptr, eps, maxiter, kmax)

cpdef uint solve_gmres_hmatrix_avector(HMatrix A, AVector b, AVector x, real eps, uint maxiter, uint kmax):
    return _krylovsolvers.solve_gmres_hmatrix_avector(<pchmatrix> A.ptr, <pcavector> b.ptr, x.ptr, eps, maxiter, kmax)