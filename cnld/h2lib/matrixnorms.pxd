## matrix_norms.pxd ##


from . cimport _matrixnorms
from . basic cimport *
from . amatrix cimport *
from . sparsematrix cimport *
from . hmatrix cimport *


cpdef real norm2diff_amatrix_sparsematrix(SparseMatrix a, AMatrix b)
cpdef real norm2diff_amatrix_hmatrix(HMatrix a, AMatrix b)
cpdef real norm2diff_sparsematrix_hmatrix(HMatrix a, SparseMatrix b)
cpdef real norm2diff_lr_amatrix(AMatrix A, AMatrix LR)
cpdef real norm2diff_lr_hmatrix(HMatrix A, HMatrix LR)
cpdef real norm2diff_lr_amatrix_hmatrix(AMatrix A, HMatrix LR)
cpdef real norm2diff_lr_hmatrix_amatrix(HMatrix A, AMatrix LR)
cpdef real norm2diff_lr_sparsematrix_amatrix(SparseMatrix A, AMatrix LR)
cpdef real norm2diff_lr_sparsematrix_hmatrix(SparseMatrix A, HMatrix LR)