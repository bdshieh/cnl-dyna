## krylovsolvers_cy.pxd ##


from . cimport _krylovsolvers
from . basic cimport *
from . amatrix cimport *
from . hmatrix cimport *


cpdef uint solve_cg_amatrix_avector(AMatrix A, AVector b, AVector x, real eps, uint maxiter)
cpdef uint solve_cg_hmatrix_avector(HMatrix A, AVector b, AVector x, real eps, uint maxiter)
cpdef uint solve_gmres_amatrix_avector(AMatrix A, AVector b, AVector x, real eps, uint maxiter, uint kmax)
cpdef uint solve_gmres_hmatrix_avector(HMatrix A, AVector b, AVector x, real eps, uint maxiter, uint kmax)