## matrixnorms.pyx ##


from . cimport _matrixnorms
from . basic cimport *
from . amatrix cimport *
from . sparsematrix cimport *
from . hmatrix cimport *


cpdef real norm2diff_amatrix_sparsematrix(SparseMatrix a, AMatrix b):
    return _matrixnorms.norm2diff_amatrix_sparsematrix(<pcsparsematrix> a.ptr, <pcamatrix> b.ptr)

cpdef real norm2diff_amatrix_hmatrix(HMatrix a, AMatrix b):
    return _matrixnorms.norm2diff_amatrix_hmatrix(<pchmatrix> a.ptr, <pcamatrix> b.ptr)

cpdef real norm2diff_sparsematrix_hmatrix(HMatrix a, SparseMatrix b):
    return _matrixnorms.norm2diff_sparsematrix_hmatrix(<pchmatrix> a.ptr, <pcsparsematrix> b.ptr)

cpdef real norm2diff_lr_amatrix(AMatrix A, AMatrix LR):
    return _matrixnorms.norm2diff_lr_amatrix(<pcamatrix> A.ptr, <pcamatrix> LR.ptr)

cpdef real norm2diff_lr_hmatrix(HMatrix A, HMatrix LR):
    return _matrixnorms.norm2diff_lr_hmatrix(<pchmatrix> A.ptr, <pchmatrix> LR.ptr)

cpdef real norm2diff_lr_amatrix_hmatrix(AMatrix A, HMatrix LR):
    return _matrixnorms.norm2diff_lr_amatrix_hmatrix(<pcamatrix> A.ptr, <pchmatrix> LR.ptr)

cpdef real norm2diff_lr_hmatrix_amatrix(HMatrix A, AMatrix LR):
    return _matrixnorms.norm2diff_lr_hmatrix_amatrix(<pchmatrix> A.ptr, <pcamatrix> LR.ptr)

cpdef real norm2diff_lr_sparsematrix_amatrix(SparseMatrix A, AMatrix LR):
    return _matrixnorms.norm2diff_lr_sparsematrix_amatrix(<pcsparsematrix> A.ptr, <pcamatrix> LR.ptr)
    
cpdef real norm2diff_lr_sparsematrix_hmatrix(SparseMatrix A, HMatrix LR):
    return _matrixnorms.norm2diff_lr_sparsematrix_hmatrix(<pcsparsematrix> A.ptr, <pchmatrix> LR.ptr)








