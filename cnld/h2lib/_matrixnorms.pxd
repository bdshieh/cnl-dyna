## matrixnorms.h ##


from . _basic cimport *
from . _amatrix cimport *
from . _sparsematrix cimport *
from . _hmatrix cimport *

cdef extern from 'matrixnorms.h' nogil:

    real norm2diff_amatrix_sparsematrix(pcsparsematrix a, pcamatrix b)
    real norm2diff_amatrix_hmatrix(pchmatrix a, pcamatrix b)
    real norm2diff_sparsematrix_hmatrix(pchmatrix a, pcsparsematrix b)
    real norm2diff_lr_amatrix(pcamatrix A, pcamatrix LR)
    real norm2diff_lr_hmatrix(pchmatrix A, pchmatrix LR)
    real norm2diff_lr_amatrix_hmatrix(pcamatrix A, pchmatrix LR)
    real norm2diff_lr_hmatrix_amatrix(pchmatrix A, pcamatrix LR)
    real norm2diff_lr_sparsematrix_amatrix(pcsparsematrix A, pcamatrix LR)
    real norm2diff_lr_sparsematrix_hmatrix(pcsparsematrix A, pchmatrix LR)

