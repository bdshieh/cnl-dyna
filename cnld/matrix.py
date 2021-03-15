'''Data-sparse (compressed) formats for matrices using H2Lib data structures.'''
__all__ = ['H2FullMatrix', 'H2SparseMatrx', 'H2HMatrix']
from timeit import default_timer as timer
import abc
import numpy as np
from matplotlib import patches
from matplotlib import pyplot as plt
from scipy.sparse import issparse
from .h2lib import *


class Matrix(abc.ABC):

    def __init__(self, mat):
        self._mat = mat

    def __del__(self):
        if self._mat is not None:
            del self._mat

    def __repr__(self):
        repr = []
        repr.append(f'{str(type(self))}\n')
        repr.append(f'  shape: {self.shape}\n')
        repr.append(f'  shape: {self.size}\n')
        repr.append(f'  nbytes: {self.nbytes / 1024 / 1024:.2f} MB\n')
        return ''.join(repr)

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
    def lu(self):
        pass

    @abc.abstractmethod
    def lusolve(self, x):
        pass

    @abc.abstractmethod
    def chol(self):
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
        if x.shape != (N,) or x.shape != (N, 1):
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
        if x.shape != (M,) or x.shape != (1, M):
            raise ValueError('Dimension mismatch')

        y = self._rmatvec(x)

        if x.ndim == 1:
            y = y.reshape(N)
        elif x.ndim == 2:
            y = y.reshape(N, 1)

        return y


class H2FullMatrix(Matrix):

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
        mat = new_zero_amatrix(shape[0], shape[1])
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
    def ndim(self):
        return 2

    @property
    def data(self):
        return np.array(self._mat.a)

    @property
    def T(self):
        B = new_zero_amatrix(*self.shape)
        copy_amatrix(True, self._mat, B)
        return H2FullMatrix(B)

    def __pos__(self):
        return super().__pos__()

    def __neg__(self):
        return super().__neg__()

    def __radd__(self, x):
        return super().__radd__(x)

    def __iadd__(self, x):
        return super().__iadd__(x)

    def __rsub__(self, x):
        return super().__rsub__(x)

    def __isub__(self, x):
        return super().__isub__(x)

    def __mul__(self, x):
        return super().__mul__(x)

    def __rmul__(self, x):
        return super().__rmul__(x)

    def __imul__(self, x):
        return super().__imul__(x)

    def __matmul__(self, x):
        return super().__matmul__(x)

    def __rmatmul__(self, x):
        return super().__rmatmul__(x)

    def __imatmul__(self, x):
        return super().__imatmul__(x)

    def __add__(self, x):

        if isinstance(x, H2FullMatrix):
            B = clone_amatrix(self._mat)
            add_amatrix(1.0, False, x._mat, B)
            return H2FullMatrix(B)

        elif isinstance(x, H2SparseMatrix):
            B = clone_amatrix(self._mat)
            add_sparsematrix_amatrix(1.0, False, x._mat, B)
            return H2FullMatrix(B)

        elif isinstance(x, H2HMatrix):
            B = clone_amatrix(self._mat)
            add_hmatrix_amatrix(1.0, False, x._mat, B)
            return H2FullMatrix(B)

        elif isinstance(x, np.ndarray):
            x = H2FullMatrix.array(x)
            B = clone_amatrix(self._mat)
            add_amatrix(1.0, False, x._mat, B)
            return H2FullMatrix(B)

        else:
            return NotImplemented

    def __sub__(self, x):

        if isinstance(x, H2FullMatrix):
            B = clone_amatrix(self._mat)
            add_amatrix(-1.0, False, x._mat, B)
            return H2FullMatrix(B)

        elif isinstance(x, H2SparseMatrix):
            B = clone_amatrix(self._mat)
            add_sparsematrix_amatrix(-1.0, False, x._mat, B)
            return H2FullMatrix(B)

        elif isinstance(x, H2HMatrix):
            B = clone_amatrix(self._mat)
            add_hmatrix_amatrix(-1.0, False, x._mat, B)
            return H2FullMatrix(B)

        elif isinstance(x, np.ndarray):
            x = H2FullMatrix.array(x)
            B = clone_amatrix(self._mat)
            add_amatrix(-1.0, False, x._mat, B)
            return H2FullMatrix(B)

        else:
            return NotImplemented

    def _smul(self, x):

        B = clone_amatrix(self._mat)
        scale_amatrix(x, B)
        return H2FullMatrix(B)

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

        if isinstance(x, H2FullMatrix):
            C = new_zero_amatrix(*self.shape)
            addmul_amatrix(1.0, False, self._mat, False, x._mat, C)
            return H2FullMatrix(C)

        elif isinstance(x, H2SparseMatrix):
            return NotImplemented

        elif isinstance(x, H2HMatrix):
            return NotImplemented

        elif isinstance(x, np.ndarray):
            x = H2FullMatrix.array(x)
            C = new_zero_amatrix(*self.shape)
            addmul_amatrix(1.0, False, self._mat, False, x._mat, C)
            return H2FullMatrix(C)
        else:
            return NotImplemented

    def _rmatmat(self, x):

        if isinstance(x, H2FullMatrix):
            C = new_zero_amatrix(*self.shape)
            addmul_amatrix(1.0, True, x._mat, True, self._mat, C)
            return H2FullMatrix(C).T

        elif isinstance(x, H2SparseMatrix):
            return NotImplemented

        elif isinstance(x, H2HMatrix):
            return NotImplemented

        elif isinstance(x, np.ndarray):
            x = H2FullMatrix.array(x)
            C = new_zero_amatrix(*self.shape)
            addmul_amatrix(1.0, True, x._mat, True, self._mat, C)
            return H2FullMatrix(C).T

        else:
            return NotImplemented

    def lu(self):
        LU = clone_amatrix(self._mat)
        succ = lrdecomp_amatrix(LU)
        if succ != 0:
            raise RuntimeError('failed to calculate LU decomposition')
        return H2FullMatrix(LU)

    def chol(self):
        CH = clone_amatrix(self._mat)
        choldecomp_amatrix(CH)
        return H2FullMatrix(CH)

    def lusolve(self, b):
        x = AVector.from_array(b)
        lrsolve_amatrix_avector(False, self._mat, x)
        return np.array(x.v)

    def cholsolve(self, b):
        x = AVector.from_array(b)
        cholsolve_amatrix_avector(self._mat, x)
        return np.array(x.v)


class H2SparseMatrix(Matrix):

    @classmethod
    def array(cls, a):

        start = timer()
        mat = SparseMatrix.from_array(a)
        time_assemble = timer() - start

        obj = cls(mat)
        obj._time_assemble = time_assemble
        return obj

    @classmethod
    def zeros(cls, shape):
        raise NotImplementedError

    @classmethod
    def ones(cls, shape):

        if len(shape) != 2:
            raise ValueError

        start = timer()
        mat = new_identity_sparsematrix(shape[0], shape[1])
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
        return getsize_sparsematrix(self._mat)

    @property
    def T(self):
        raise NotImplementedError

    @property
    def ndim(self):
        return 2

    @property
    def data(self):
        raise NotImplementedError

    def __pos__(self):
        return super().__pos__()

    def __neg__(self):
        return super().__neg__()

    def __radd__(self, x):
        return super().__radd__(x)

    def __iadd__(self, x):
        return super().__iadd__(x)

    def __rsub__(self, x):
        return super().__rsub__(x)

    def __isub__(self, x):
        return super().__isub__(x)

    def __mul__(self, x):
        return super().__mul__(x)

    def __rmul__(self, x):
        return super().__rmul__(x)

    def __imul__(self, x):
        return super().__imul__(x)

    def __matmul__(self, x):
        return super().__matmul__(x)

    def __rmatmul__(self, x):
        return super().__rmatmul__(x)

    def __imatmul__(self, x):
        return super().__imatmul__(x)

    def __add__(self, x):

        if isinstance(x, H2FullMatrix):
            B = clone_amatrix(x._mat)
            add_sparsematrix_amatrix(1.0, False, self._mat, B)
            return H2FullMatrix(B)

        elif isinstance(x, H2SparseMatrix):
            return NotImplemented

        elif isinstance(x, H2HMatrix):
            return NotImplemented

        elif isinstance(x, np.ndarray):
            x = H2FullMatrix.array(x)
            B = clone_amatrix(x._mat)
            add_amatrix(1.0, False, self._mat, B)
            return H2FullMatrix(B)

        else:
            return NotImplemented

    def __sub__(self, x):

        if isinstance(x, H2FullMatrix):
            B = clone_amatrix(x._mat)
            add_amatrix(-1.0, False, self._mat, B)
            return H2FullMatrix(B)

        elif isinstance(x, H2SparseMatrix):
            return NotImplemented

        elif isinstance(x, H2HMatrix):
            return NotImplemented

        elif isinstance(x, np.ndarray):
            x = H2FullMatrix.array(x)
            B = clone_amatrix(x._mat)
            add_amatrix(-1.0, False, self._mat, B)
            return -1 * H2FullMatrix(B)

        else:
            return NotImplemented

    def _smul(self, x):
        return NotImplemented

    def _matvec(self, x):

        xv = AVector.from_array(x)
        y = AVector(x.size)
        clear_avector(y)
        addeval_sparsematrix_avector(1.0, self._mat, xv, y)
        return np.array(y.v)

    def _rmatvec(self, x):

        xv = AVector.from_array(x)
        y = AVector(x.size)
        clear_avector(y)
        addevaltrans_amatrix_avector(1.0, self._mat, xv, y)
        return np.array(y.v)

    def _matmat(self, x):

        if isinstance(x, H2FullMatrix):
            return NotImplemented

        elif isinstance(x, H2SparseMatrix):
            return NotImplemented

        elif isinstance(x, H2HMatrix):
            return NotImplemented

        elif isinstance(x, np.ndarray):
            return NotImplemented

        else:
            return NotImplemented

    def _rmatmat(self, x):

        if isinstance(x, H2FullMatrix):
            return NotImplemented

        elif isinstance(x, H2SparseMatrix):
            return NotImplemented

        elif isinstance(x, H2HMatrix):
            return NotImplemented

        elif isinstance(x, np.ndarray):
            return NotImplemented

        else:
            return NotImplemented

    def lu(self):
        raise NotImplementedError

    def chol(self):
        raise NotImplementedError

    def lusolve(self, b):
        raise NotImplementedError

    def cholsolve(self, b):
        raise NotImplementedError


class H2HMatrix(Matrix):

    eps_add = 1e-12

    def __init__(self, mat, root, broot):
        self._mat = mat
        self._root = root
        self._broot = broot

    def __del__(self):
        super(self)
        del self._root
        del self._broot

    @classmethod
    def array(cls, a):
        raise NotImplementedError

    @classmethod
    def zeros(cls, shape):
        raise NotImplementedError

    @property
    def time_assemble(self):
        return self._time_assemble

    @property
    def shape(self):
        return getrows_hmatrix(self._mat), getcols_hmatrix(self._mat)

    @property
    def size(self):
        return getrows_hmatrix(self._mat) * getcols_hmatrix(self._mat)

    @property
    def nbytes(self):
        return getsize_hmatrix(self._mat)

    @property
    def T(self):
        raise NotImplementedError

    @property
    def ndim(self):
        return 2

    @property
    def data(self):
        raise NotImplementedError

    def __pos__(self):
        return super().__pos__()

    def __neg__(self):
        return super().__neg__()

    def __radd__(self, x):
        return super().__radd__(x)

    def __iadd__(self, x):
        return super().__iadd__(x)

    def __rsub__(self, x):
        return super().__rsub__(x)

    def __isub__(self, x):
        return super().__isub__(x)

    def __mul__(self, x):
        return super().__mul__(x)

    def __rmul__(self, x):
        return super().__rmul__(x)

    def __imul__(self, x):
        return super().__imul__(x)

    def __matmul__(self, x):
        return super().__matmul__(x)

    def __rmatmul__(self, x):
        return super().__rmatmul__(x)

    def __imatmul__(self, x):
        return super().__imatmul__(x)

    def __add__(self, x):

        if isinstance(x, H2FullMatrix):
            B = clone_hmatrix(self._mat)
            tm = new_releucl_truncmode()
            add_amatrix_hmatrix(1.0, False, x._mat, tm, self.eps_add, B)
            return H2HMatrix(B)

        elif isinstance(x, H2SparseMatrix):
            B = clone_hmatrix(self._mat)
            tm = new_releucl_truncmode()

            # sparse format is converted to hformat prior to addition
            hm = clonestructure_hmatrix(self._mat)
            clear_hmatrix(hm)  # very important to clear hmatrix
            copy_sparsematrix_hmatrix(x._mat, hm)

            add_hmatrix(1.0, hm, tm, self.eps_add, B)
            return H2HMatrix(B)

        elif isinstance(x, H2HMatrix):
            B = clone_hmatrix(self._mat)
            tm = new_releucl_truncmode()
            add_hmatrix(1.0, x._mat, tm, self.eps_add, B)
            return H2HMatrix(B)

        elif isinstance(x, np.ndarray):
            x = H2FullMatrix.array(x)
            B = clone_hmatrix(self._mat)
            tm = new_releucl_truncmode()
            add_amatrix_hmatrix(1.0, False, x._mat, tm, self.eps_add, B)
            return H2HMatrix(B)

        else:
            return NotImplemented

    def __sub__(self, x):

        if isinstance(x, H2FullMatrix):
            B = clone_hmatrix(self._mat)
            tm = new_releucl_truncmode()
            add_amatrix_hmatrix(-1.0, False, x._mat, tm, self.eps_add, B)
            return H2HMatrix(B)

        elif isinstance(x, H2SparseMatrix):
            B = clone_hmatrix(self._mat)
            tm = new_releucl_truncmode()

            # sparse format is converted to hformat prior to addition
            hm = clonestructure_hmatrix(self._mat)
            clear_hmatrix(hm)  # very important to clear hmatrix
            copy_sparsematrix_hmatrix(x._mat, hm)

            add_hmatrix(-1.0, hm, tm, self.eps_add, B)
            return H2HMatrix(B)

        elif isinstance(x, H2HMatrix):
            B = clone_hmatrix(self._mat)
            tm = new_releucl_truncmode()
            add_hmatrix(-1.0, x._mat, tm, self.eps_add, B)
            return H2HMatrix(B)

        elif isinstance(x, np.ndarray):
            x = H2FullMatrix.array(x)
            B = clone_hmatrix(self._mat)
            tm = new_releucl_truncmode()
            add_amatrix_hmatrix(-1.0, False, x._mat, tm, self.eps_add, B)
            return H2HMatrix(B)

        else:
            return NotImplemented

    def _smul(self, x):

        id = clonestructure_hmatrix(self._mat)
        identity_hmatrix(id)
        z = clonestructure_hmatrix(self._mat)
        clear_hmatrix(z)
        tm = new_releucl_truncmode()
        addmul_hmatrix(x, False, id, False, self._mat, tm, self.eps_add, z)
        return HFormat(z)

    def _matvec(self, x):

        xv = AVector.from_array(x)
        y = AVector(x.size)
        clear_avector(y)
        addeval_hmatrix_avector(1.0, self._mat, xv, y)
        return np.array(y.v)

    def _rmatvec(self, x):

        xv = AVector.from_array(x)
        y = AVector(x.size)
        clear_avector(y)
        addeval_amatrix_avector(1.0, (self.T)._mat, xv, y)
        return np.array(y.v)

    def _matmat(self, x):

        if isinstance(x, H2FullMatrix):
            return NotImplemented

        elif isinstance(x, H2SparseMatrix):
            return NotImplemented

        elif isinstance(x, H2HMatrix):
            return NotImplemented

        elif isinstance(x, np.ndarray):
            C = clonestructure_hmatrix(self._mat)
            clear_hmatrix(C)
            tm = new_releucl_truncmode()
            addmul_hmatrix(1.0, False, x._mat, False, self._mat, tm,
                           self.eps_add, C)
            return HFormat(C)

        else:
            return NotImplemented

    def _rmatmat(self, x):

        if isinstance(x, H2FullMatrix):
            C = new_zero_amatrix(*self.shape)
            addmul_amatrix(1.0, True, x._mat, True, self._mat, C)
            return H2FullMatrix(C).T

        elif isinstance(x, H2SparseMatrix):
            return NotImplemented

        elif isinstance(x, H2HMatrix):
            return NotImplemented

        elif isinstance(x, np.ndarray):
            x = H2FullMatrix.array(x)
            C = new_zero_amatrix(*self.shape)
            addmul_amatrix(1.0, True, x._mat, True, self._mat, C)
            return H2FullMatrix(C).T

        else:
            return NotImplemented

    def lu(self, eps=1e-12):
        LU = clone_hmatrix(self._mat)
        tm = new_releucl_truncmode()
        lrdecomp_hmatrix(LU, tm, eps)
        return HFormat(LU)

    def chol(self, eps=1e-12):
        CHOL = clone_hmatrix(self._mat)
        tm = new_releucl_truncmode()
        choldecomp_hmatrix(CHOL, tm, eps)
        return HFormat(CHOL)

    def lusolve(self, b):
        x = AVector.from_array(b)
        lrsolve_hmatrix_avector(False, self._mat, x)
        return np.array(x.v)

    def cholsolve(self, b):
        x = AVector.from_array(b)
        cholsolve_hmatrix_avector(self._mat, x)
        return np.array(x.v)

    def _draw_hmatrix(self, hm, bbox, maxidx, ax):
        if len(hm.son) == 0:
            if hm.r:
                rk = str(hm.r.k)
                fill = False
            elif hm.f:
                rk = None
                fill = True
            else:
                raise Exception

            x0, y0, x1, y1 = bbox
            width, height = x1 - x0, y1 - y0
            sq = patches.Rectangle((x0, y0),
                                   width,
                                   height,
                                   edgecolor='black',
                                   fill=fill,
                                   facecolor='black')
            ax.add_patch(sq)
            if rk:
                fontsize = int(round((112 - 6) * width + 6))
                if width > 0.03:
                    ax.text(x0 + 0.05 * width,
                            y0 + 0.95 * height,
                            rk,
                            fontsize=fontsize)

        else:
            rmax, cmax = maxidx
            x0, y0, x1, y1 = bbox

            rsidx = (0, 1, 0, 1)
            csidx = (0, 0, 1, 1)

            width0 = len(hm.son[0].cc.idx) / cmax
            height0 = len(hm.son[0].rc.idx) / rmax

            for i, s in enumerate(hm.son):
                width = len(s.cc.idx) / cmax
                height = len(s.rc.idx) / rmax

                xnew = x0 if csidx[i] == 0 else x0 + width0
                ynew = y0 if rsidx[i] == 0 else y0 + height0

                if csidx[i] == 0:
                    xnew = x0
                else:
                    xnew = x0 + width0
                if rsidx[i] == 0:
                    ynew = y0
                else:
                    ynew = y0 + height0

                bbox = xnew, ynew, xnew + width, ynew + height
                self._draw_hmatrix(s, bbox, maxidx, ax)

    def draw(self):
        hm = self._mat
        maxidx = len(hm.rc.idx), len(hm.cc.idx)

        fig, ax = plt.subplots(figsize=(9, 9))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.invert_yaxis()
        ax.set_aspect('equal')
        self._draw_hmatrix(hm, (0, 0, 1, 1), maxidx, ax)
        fig.show()