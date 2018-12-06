## krylovsolvers_cy.pxd ##


from . cimport krylovsolvers as _krylovsolvers
from . basic_cy cimport *
from . amatrix_cy cimport *
from . hmatrix_cy cimport *


cpdef uint solve_cg_amatrix_avector(AMatrix A, AVector b, AVector x, real eps, uint maxiter)
cpdef uint solve_cg_hmatrix_avector(HMatrix A, AVector b, AVector x, real eps, uint maxiter)
cpdef uint solve_gmres_amatrix_avector(AMatrix A, AVector b, AVector x, real eps, uint maxiter, uint kmax)
cpdef uint solve_gmres_hmatrix_avector(HMatrix A, AVector b, AVector x, real eps, uint maxiter, uint kmax)