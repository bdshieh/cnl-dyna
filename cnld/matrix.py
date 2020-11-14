'''Data-sparse (compressed) formats for matrices using H2Lib data structures.'''
from timeit import default_timer as timer
import abc

import numpy as np
from matplotlib import patches
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix, issparse

from .h2lib import *


class Matrix(abc.ABC):
    def __init__(self, mat):
        self._mat = mat

    def __del__(self):
        if self._mat is not None:
            del self._mat

    @abc.abstractproperty
    def shape(self):
        pass

    @abc.abstractproperty
    def size(self):
        pass

    @abc.abstractproperty
    def ndim(self):
        return 2

    @abc.abstractproperty
    def nbytes(self):
        pass

    @abc.abstractproperty
    def T(self):
        pass

    @abc.abstractproperty
    def data(self):
        return np.array(self._mat.a)

    @abc.abstractmethod
    def __pos__(self):
        return self

    @abc.abstractmethod
    def __neg__(self):
        return -1 * self

    @abc.abstractmethod
    def __add__(self, x):
        pass

    @abc.abstractmethod
    def __radd__(self, x):
        return self.__add__(x)

    @abc.abstractmethod
    def __iadd__(self, x):
        pass

    @abc.abstractmethod
    def __iadd__(self, x):
        return self._add__(x)

    @abc.abstractmethod
    def __rsub__(self, x):
        return -1 * self.__sub__(x)

    @abc.abstractmethod
    def __isub__(self, x):
        return self.__sub__(x)

    @abc.abstractmethod
    def __mul__(self, x):
        if np.isscalar(x):
            return self._smul(x)

        if x.ndim == 1 or (x.ndim == 2 and x.shape[1] == 1):
            return self.matvec(x)
        elif x.ndim == 2:
            return NotImplemented
        else:
            return NotImplemented

    @abc.abstractmethod
    def __rmul__(self, x):
        if np.isscalar(x):
            return self._smul(x)

        if x.ndim == 1 or (x.ndim == 2 and x.shape[0] == 1):
            return self.rmatvec(x)
        elif x.ndim == 2:
            return NotImplemented
        else:
            return NotImplemented

    @abc.abstractmethod
    def __imul__(self, x):
        return self.__mul__(x)

    @abc.abstractmethod
    def __matmul__(self, x):
        if np.isscalar(x):
            return self._smul(x)

        if x.ndim == 1 or (x.ndim == 2 and x.shape[1] == 1):
            return self.matvec(x)
        elif x.ndim == 2:
            return self.matmat(x)
        else:
            raise ValueError

    @abc.abstractmethod
    def __rmatmul__(self, x):
        if np.isscalar(x):
            return self._smul(x)

        if x.ndim == 1 or (x.ndim == 2 and x.shape[1] == 1):
            return self.rmatvec(x)
        elif x.ndim == 2:
            return self.rmatmat(x)
        else:
            raise ValueError

    @abc.abstractmethod
    def __imatmul__(self, x):
        return self.__matmul__(x)

    @abc.abstractmethod
    def lu(self, x):
        pass

    @abc.abstractmethod
    def lusolve(self, x):
        pass

    @abc.abstractmethod
    def chol(self, x):
        pass

    @abc.abstractmethod
    def cholsolve(self, x):
        pass

    @abc.abstractmethod
    def _smul(self, x):
        pass

    @abc.abstractmethod
    def _matvec(self, x):
        pass

    @abc.abstractmethod
    def _rmatvec(self, x):
        pass

    @abc.abstractmethod
    def _matmat(self, x):
        pass

    @abc.abstractmethod
    def _rmatmat(self, x):
        pass

    def matmat(self, x):

        if x.ndim != 2:
            raise ValueError
        M, N = self.shape
        if x.shape[0] != N:
            raise ValueError

        Y = self._matmat(x)

        return Y

    def matvec(self, x):

        M, N = self.shape
        if x.shape != (N, ) or x.shape != (N, 1):
            raise ValueError('Dimension mismatch')

        y = self._matvec(x)

        if x.ndim == 1:
            y = y.reshape(M)
        elif x.ndim == 2:
            y = y.reshape(M, 1)

        return y

    def rmatmat(self, x):

        if x.ndim != 2:
            raise ValueError
        M, N = self.shape
        if x.shape[1] != M:
            raise ValueError

        Y = self._rmatmat(x)

        return Y

    def rmatvec(self, x):

        M, N = self.shape
        if x.shape != (M, ) or x.shape != (1, M):
            raise ValueError('Dimension mismatch')

        y = self._rmatvec(x)

        if x.ndim == 1:
            y = y.reshape(N)
        elif x.ndim == 2:
            y = y.reshape(N, 1)

        return y


class FullMatrix(Matrix):
    @classmethod
    def array(cls, a):

        if issparse(a):
            a = a.toarray()

        start = timer()
        mat = AMatrix.from_array(a)
        time_assemble = timer() - start

        obj = cls(mat)
        obj._time_assemble = time_assemble
        return obj

    @classmethod
    def zeros(cls, shape):

        if len(shape) != 2:
            raise ValueError

        start = timer()
        mat = AMatrix.new_zero_amatrix(shape[0], shape[1])
        time_assemble = timer() - start

        obj = cls(mat)
        obj._time_assemble = time_assemble
        return obj

    @property
    def time_assemble(self):
        return self._time_assemble

    @property
    def shape(self):
        return self._mat.rows, self._mat.cols

    @property
    def size(self):
        return self._mat.rows * self._mat.cols

    @property
    def nbytes(self):
        return getsize_amatrix(self._mat)

    @property
    def T(self):
        B = new_zero_amatrix(*self.shape)
        copy_amatrix(True, self._mat, B)
        return FullMatrix(B)

    def __add__(self, x):

        if isinstance(x, FullMatrix):
            B = clone_amatrix(self._mat)
            add_amatrix(1.0, False, x._mat, B)
            return FullMatrix(B)

        elif isinstance(x, SparseMatrix):
            B = clone_amatrix(self._mat)
            add_sparsematrix_amatrix(1.0, False, x._mat, B)
            return FullMatrix(B)

        elif isinstance(x, HMatrix):
            B = clone_amatrix(self._mat)
            add_hmatrix_amatrix(1.0, False, x._mat, B)
            return FullMatrix(B)

        elif isinstance(x, np.ndarray):
            x = FullMatrix.array(x)
            B = clone_amatrix(self._mat)
            add_amatrix(1.0, False, x._mat, B)
            return FullMatrix(B)

        else:
            return NotImplemented

    def __sub__(self, x):

        if isinstance(x, FullMatrix):
            B = clone_amatrix(self._mat)
            add_amatrix(-1.0, False, x._mat, B)
            return FullMatrix(B)

        elif isinstance(x, SparseMatrix):
            B = clone_amatrix(self._mat)
            add_sparsematrix_amatrix(-1.0, False, x._mat, B)
            return FullMatrix(B)

        elif isinstance(x, HMatrix):
            B = clone_amatrix(self._mat)
            add_hmatrix_amatrix(-1.0, False, x._mat, B)
            return FullMatrix(B)

        elif isinstance(x, np.ndarray):
            x = FullMatrix.array(x)
            B = clone_amatrix(self._mat)
            add_amatrix(-1.0, False, x._mat, B)
            return FullMatrix(B)

        else:
            return NotImplemented

    def _smul(self, x):

        B = clone_amatrix(self._mat)
        scale_amatrix(x, B)
        return FullMatrix(B)

    def _matvec(self, x):

        xv = AVector.from_array(x)
        y = AVector(x.size)
        clear_avector(y)
        addeval_amatrix_avector(1.0, self._mat, xv, y)
        return np.array(y.v)

    def _rmatvec(self, x):

        xv = AVector.from_array(x)
        y = AVector(x.size)
        clear_avector(y)
        addeval_amatrix_avector(1.0, (self.T)._mat, xv, y)
        return np.array(y.v)

    def _matmat(self, x):

        if isinstance(x, FullMatrix):
            C = new_zero_amatrix(*self.shape)
            addmul_amatrix(1.0, False, self._mat, False, x._mat, C)
            return FullMatrix(C)

        elif isinstance(x, SparseMatrix):
            return NotImplemented

        elif isinstance(x, HMatrix):
            return NotImplemented

        elif isinstance(x, np.ndarray):
            x = FullMatrix.array(x)
            C = new_zero_amatrix(*self.shape)
            addmul_amatrix(1.0, False, self._mat, False, x._mat, C)
            return FullMatrix(C)
        else:
            return NotImplemented

    def _rmatmat(self, x):

        if isinstance(x, FullMatrix):
            C = new_zero_amatrix(*self.shape)
            addmul_amatrix(1.0, True, x._mat, True, self._mat, C)
            return FullMatrix(C).T

        elif isinstance(x, SparseMatrix):
            return NotImplemented

        elif isinstance(x, HMatrix):
            return NotImplemented

        elif isinstance(x, np.ndarray):
            x = FullMatrix.array(x)
            C = new_zero_amatrix(*self.shape)
            addmul_amatrix(1.0, True, x._mat, True, self._mat, C)
            return FullMatrix(C).T

        else:
            return NotImplemented

    def lu(self):
        LU = clone_amatrix(self._mat)
        succ = lrdecomp_amatrix(LU)
        if succ != 0:
            raise RuntimeError('failed to calculate LU decomposition')
        return FullMatrix(LU)

    def chol(self):
        CH = clone_amatrix(self._mat)
        choldecomp_amatrix(CH)
        return FullMatrix(CH)

    def lusolve(self, b):
        x = AVector.from_array(b)
        lrsolve_amatrix_avector(False, self._mat, x)
        return np.array(x.v)

    def cholsolve(self, b):
        x = AVector.from_array(b)
        cholsolve_amatrix_avector(self._mat, x)
        return np.array(x.v)


class SparseMatrix(BaseMatrix):
    pass


class HMatrix(BaseMatrix):
    pass
