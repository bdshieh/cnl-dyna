## krylovsolvers.h ##


from . _basic cimport *
from . _amatrix cimport *
from . _hmatrix cimport *

cdef extern from 'krylovsolvers.h' nogil:

    uint solve_cg_amatrix_avector(pcamatrix A, pcavector b, pavector x, real eps, uint maxiter)
    uint solve_cg_hmatrix_avector(pchmatrix A, pcavector b, pavector x, real eps, uint maxiter)
    uint solve_gmres_amatrix_avector(pcamatrix A, pcavector b, pavector x, real eps, uint maxiter, uint kmax)
    uint solve_gmres_hmatrix_avector(pchmatrix A, pcavector b, pavector x, real eps, uint maxiter, uint kmax)