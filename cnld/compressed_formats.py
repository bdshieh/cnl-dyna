'''
Data-sparse (compressed) formats for matrices using h2lib
'''
from timeit import default_timer as timer

import numpy as np
from matplotlib import patches
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix, issparse

from .h2lib import *


class BaseFormat:
    '''
    Base class defining abstract interface for formats.
    '''
    def __init__(self, mat):
        self._mat = mat

    def __del__(self):
        if self._mat is not None:
            del self._mat

    ''' PROPERTIES '''
    @property
    def rows(self):
        return

    @property
    def cols(self):
        return

    @property
    def shape(self):
        return self.rows, self.cols

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def format(self):
        return self.__class__.__name__

    ''' MAGIC OPERATIONS '''
    def _add(self, x):
        return NotImplemented

    def __add__(self, x):
        if not isinstance(x, BaseFormat):
            raise ValueError('operation not supported with this type')

        if self.shape != x.shape:
            raise ValueError('dimension mismatch')

        return self._add(x)

    def __radd__(self, x):
        return self.__add__(x)

    def __rmul__(self, x):
        if not np.isscalar(x):
            return NotImplemented
        return self.__mul__(x)

    def __mul__(self, x):
        return self.dot(x)

    def __call__(self, x):
        return self * x

    def __neg__(self):
        return self * -1

    def __sub__(self, x):
        return self.__add__(-x)

    def __rsub__(self, x):
        return self.__sub__(x) * -1

    ''' LINALG OPERATIONS '''
    def _smul(self, x):
        raise NotImplementedError

    def _matmat(self, x):
        return NotImplemented

    def _matvec(self, x):
        return NotImplemented

    def matmat(self, X):
        # X = np.asanyarray(X)
        if X.ndim != 2:
            raise ValueError
        M, N = self.shape
        if X.shape[0] != N:
            raise ValueError

        Y = self._matmat(X)

        return Y

    def matvec(self, x):
        # x = np.asanyarray(x)
        M, N = self.shape
        if x.shape != (N, ) and x.shape != (N, 1):
            raise ValueError('dimension mismatch')

        y = self._matvec(x)

        if x.ndim == 1:
            y = y.reshape(M)
        elif x.ndim == 2:
            y = y.reshape(M, 1)

        return y

    def dot(self, x):
        if np.isscalar(x):
            return self._smul(x)

        # # convert all numpy arrays to h2lib arrays
        # elif isinstance(x, np.ndarray):
        #     if x.ndim == 1 or x.ndim == 2 and x.shape[1] == 1:
        #         xv = AVector.from_array(x)
        #     else:
        #         xv = AMatrix.from_array(x)

        if x.ndim == 1 or x.ndim == 2 and x.shape[1] == 1:
            return self.matvec(x)
        elif x.ndim == 2:
            return self.matmat(x)
        else:
            raise ValueError

    def _adjoint(self):
        return NotImplemented

    def _transpose(self):
        return NotImplemented

    def adjoint(self):
        return self._adjoint()

    def transpose(self):
        return self._transpose()

    ''' LINALG SOLVING '''
    def _lu(self):
        raise NotImplementedError

    def _chol(self):
        raise NotImplementedError

    def _lusolve(self, b):
        raise NotImplementedError

    def _cholsolve(self, b):
        raise NotImplementedError

    def lu(self):
        return self._lu()

    def lusolve(self, b):
        return self._lusolve(b)

    def chol(self):
        return self._chol()

    def cholsolve(self, b):
        return self._cholsolve(b)


class FullFormat(BaseFormat):
    '''
    Full (dense) matrix format, i.e. no compression
    '''
    ''' PROPERTIES '''
    @property
    def rows(self):
        return self._mat.rows

    @property
    def cols(self):
        return self._mat.cols

    @property
    def size(self):
        return getsize_amatrix(self._mat)

    @property
    def data(self):
        return np.array(self._mat.a)

    ''' INDEXING '''
    def __getitem__(self, key):
        return self._mat.a[key]

    def __setitem__(self, key, val):
        self._mat.a[key] = val

    ''' OPERATIONS '''
    def _add(self, x):
        if isinstance(x, FullFormat):
            B = clone_amatrix(self._mat)
            add_amatrix(1.0, False, x._mat, B)
            return FullFormat(B)
        elif isinstance(x, SparseFormat):
            B = clone_amatrix(self._mat)
            add_sparsematrix_amatrix(1.0, False, x._mat, B)
            return FullFormat(B)
        elif isinstance(x, HFormat):
            B = clone_amatrix(self._mat)
            add_hmatrix_amatrix(1.0, False, x._mat, B)
            return FullFormat(B)
        else:
            return NotImplemented

    def _smul(self, x):
        B = clone_amatrix(self._mat)
        scale_amatrix(x, B)
        return FullFormat(B)

    # def _matmat(self, x):
    #     if isinstance(x, FullFormat):
    #         # B = clone_amatrix(self._mat)
    #         C = new_zero_amatrix(*self.shape)
    #         addmul_amatrix(1.0, False, self._mat, False, x._mat, C)
    #         return FullFormat(C)
    #     elif isinstance(x, SparseFormat):
    #         raise NotImplementedError('operation not supported with this type')
    #     elif isinstance(x, HFormat):
    #         raise NotImplementedError('operation not supported with this type')
    #     else:
    #         raise ValueError('operation with unrecognized type')

    def _matvec(self, x):
        xv = AVector.from_array(x)
        y = AVector(x.size)
        clear_avector(y)
        addeval_amatrix_avector(1.0, self._mat, xv, y)
        # addevalsymm_hmatrix_avector(1.0, self._mat, x, y)
        out = np.array(y.v)
        return out

    def _lu(self):
        LU = clone_amatrix(self._mat)
        succ = lrdecomp_amatrix(LU)
        if succ != 0:
            raise RuntimeError('failed to calculate LU decomposition')
        return FullFormat(LU)

    def _chol(self):
        CH = clone_amatrix(self._mat)
        choldecomp_amatrix(CH)
        return FullFormat(CH)

    def _lusolve(self, b):
        x = AVector.from_array(b)
        lrsolve_amatrix_avector(False, self._mat, x)
        return np.array(x.v)

    def _cholsolve(self, b):
        x = AVector.from_array(b)
        cholsolve_amatrix_avector(self._mat, x)
        return np.array(x.v)

    def _triangularsolve(self, b):
        x = AVector.from_array(b)
        lrsolve_amatrix_avector(False, self._mat, x)
        # triangularsolve_amatrix_avector(True, False, True, self._mat, x)
        # triangularsolve_amatrix_avector(False, False, False, self._mat, x)
        return np.array(x.v)


class SparseFormat(BaseFormat):
    '''
    Sparse matrix format
    '''
    ''' PROPERTIES '''
    @property
    def rows(self):
        return self._mat.rows

    @property
    def cols(self):
        return self._mat.cols

    @property
    def size(self):
        return getsize_sparsematrix(self._mat)

    @property
    def nnz(self):
        return self._mat.nz

    @property
    def row(self):
        return self._mat.row

    @property
    def col(self):
        return self._mat.col

    @property
    def coeff(self):
        return self._mat.coeff

    ''' OPERATIONS '''
    def _add(self, x):
        return NotImplemented

    def _smul(self, x):
        raise NotImplementedError('operation not supported with this type')

    def _matmat(self, x):
        if isinstance(x, FullFormat):
            raise NotImplementedError('operation not supported with this type')
        elif isinstance(x, SparseFormat):
            raise NotImplementedError('operation not supported with this type')
        elif isinstance(x, HFormat):
            raise NotImplementedError('operation not supported with this type')
        else:
            raise ValueError('operation with unrecognized type')

    def _matvec(self, x):
        xv = AVector.from_array(x)
        y = AVector(x.size)
        clear_avector(y)
        addeval_sparsematrix_avector(1.0, self._mat, xv, y)
        return np.array(y.v)

    def _lu(self):
        raise NotImplementedError('operation not supported with this type')

    def _chol(self):
        raise NotImplementedError('operation not supported with this type')

    def _lusolve(self, b):
        raise NotImplementedError('operation not supported with this type')

    def _cholsolve(self, b):
        raise NotImplementedError('operation not supported with this type')

    ''' OTHER '''
    def _as_hformat(self, href):
        '''
        Convert sparse format to hierarchical format using
        the h-structure in href
        '''
        hm = clonestructure_hmatrix(href)
        clear_hmatrix(
            hm
        )  # very important to clear hmatrix otherwise addition doesn't work properly
        copy_sparsematrix_hmatrix(self._mat, hm)
        return HFormat(hm)


class HFormat(BaseFormat):
    '''
    Hierarchical matrix format
    '''
    ''' DATA ATTRIBUTES '''
    eps_add = 1e-12
    eps_lu = 1e-12
    eps_chol = 1e-12
    ''' PROPERTIES '''
    @property
    def rows(self):
        return getrows_hmatrix(self._mat)

    @property
    def cols(self):
        return getcols_hmatrix(self._mat)

    @property
    def size(self):
        return getsize_hmatrix(self._mat)

    ''' OPERATIONS '''
    def _add(self, x):
        if isinstance(x, FullFormat):
            B = clone_hmatrix(self._mat)
            tm = new_releucl_truncmode()
            add_amatrix_hmatrix(1.0, False, x._mat, tm, self.eps_add, B)
            return HFormat(B)
        elif isinstance(x, SparseFormat):
            B = clone_hmatrix(self._mat)
            tm = new_releucl_truncmode()
            # sparse format is converted to hformat prior to addition
            add_hmatrix(1, (x._as_hformat(self._mat))._mat, tm, self.eps_add,
                        B)
            return HFormat(B)
        elif isinstance(x, HFormat):
            B = clone_hmatrix(self._mat)
            tm = new_releucl_truncmode()
            add_hmatrix(1, x._mat, tm, self.eps_add, B)
            return HFormat(B)
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

    def _matmat(self, x):
        if isinstance(x, FullFormat):
            raise NotImplementedError('operation not supported with this type')
        elif isinstance(x, SparseFormat):
            raise NotImplementedError('operation not supported with this type')
        elif isinstance(x, HFormat):
            C = clonestructure_hmatrix(self._mat)
            clear_hmatrix(C)
            tm = new_releucl_truncmode()
            addmul_hmatrix(1.0, False, x._mat, False, self._mat, tm,
                           self.eps_add, C)
            return HFormat(C)
        else:
            raise ValueError('operation with unrecognized type')

    def _matvec(self, x):
        xv = AVector.from_array(x)
        y = AVector(x.size)
        clear_avector(y)
        addeval_hmatrix_avector(1.0, self._mat, xv, y)
        # addevalsymm_hmatrix_avector(1.0, self._mat, x, y)
        return np.array(y.v)

    def _lu(self):
        LU = clone_hmatrix(self._mat)
        tm = new_releucl_truncmode()
        lrdecomp_hmatrix(LU, tm, self.eps_lu)
        return HFormat(LU)

    def _chol(self):
        CHOL = clone_hmatrix(self._mat)
        tm = new_releucl_truncmode()
        choldecomp_hmatrix(CHOL, tm, self.eps_chol)
        return HFormat(CHOL)

    def _lusolve(self, b):
        x = AVector.from_array(b)
        lrsolve_hmatrix_avector(False, self._mat, x)
        return np.array(x.v)

    def _cholsolve(self, b):
        x = AVector.from_array(b)
        cholsolve_hmatrix_avector(self._mat, x)
        return np.array(x.v)

    def _triangularsolve(self, b):
        x = AVector.from_array(b)
        lrsolve_hmatrix_avector(False, self._mat, x)
        # triangularsolve_hmatrix_avector(True, False, False, self._mat, x)
        # triangularsolve_hmatrix_avector(False, False, False, self._mat, x)
        return np.array(x.v)

    ''' OTHER '''
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


def lu(A, eps=1e-12):
    A.eps_lu = eps
    return A.lu()


def chol(A, eps=1e-12):
    A.eps_chol = eps
    return A.chol()


def lusolve(A, b):
    return A.lusolve(b)


def cholsolve(A, b):
    return A.cholsolve(b)


def _mbk_repr(self):

    repr = []
    repr.append('MBKMatrix (Mass, Damping, Stiffness Matrix)\n')
    repr.append(f'  BaseFormat: {self.format}\n')
    repr.append(f'  Shape: {self.shape}\n')
    repr.append(f'  Size: {self.size / 1024 / 1024:.2f} MB\n')
    return ''.join(repr)


def _z_repr(self):

    repr = []
    repr.append('ZMatrix (Acoustic Impedance Matrix)\n')
    repr.append(f'  BaseFormat: {self.format}\n')
    repr.append(f'  Shape: {self.shape}\n')
    repr.append(f'  Size: {self.size / 1024 / 1024:.2f} MB\n')
    return ''.join(repr)


class MbkFullMatrix(FullFormat):
    def __init__(self, array):

        if issparse(array):
            array = array.toarray()

        start = timer()
        MBK = AMatrix.from_array(array)
        time_assemble = timer() - start

        self._mat = MBK
        self._time_assemble = time_assemble

    @property
    def time_assemble(self):
        return self._time_assemble

    __repr__ = _mbk_repr


class MbkSparseMatrix(SparseFormat):
    def __init__(self, array):

        array = csr_matrix(array)

        start = timer()
        MBK = SparseMatrix.from_array(array)
        time_assemble = timer() - start

        self._mat = MBK
        self._time_assemble = time_assemble

    @property
    def time_assemble(self):
        return self._time_assemble

    __repr__ = _mbk_repr


class ZFullMatrix(FullFormat):
    def __init__(self, mesh, k, basis='linear', q_reg=2, q_sing=4, **kwargs):

        if basis.lower() in ['constant']:
            _basis = basisfunctionbem3d.CONSTANT
        elif basis.lower() in ['linear']:
            _basis = basisfunctionbem3d.LINEAR
        else:
            raise TypeError

        bem = new_slp_helmholtz_bem3d(k, mesh.surface3d, q_reg, q_sing, _basis,
                                      _basis)

        Z = AMatrix(len(mesh.vertices), len(mesh.vertices))

        start = timer()
        assemble_bem3d_amatrix(bem, Z)
        time_assemble = timer() - start

        self._mat = Z
        self._time_assemble = time_assemble
        # self._bem = bem

    @property
    def time_assemble(self):
        return self._time_assemble

    __repr__ = _z_repr


class ZHMatrix(HFormat):
    def __init__(self,
                 mesh,
                 k,
                 basis='linear',
                 m=4,
                 q_reg=2,
                 q_sing=4,
                 aprx='paca',
                 admis='2',
                 eta=1.0,
                 eps_aca=1e-2,
                 strict=False,
                 clf=16,
                 rk=0,
                 **kwargs):

        if basis.lower() in ['constant']:
            _basis = basisfunctionbem3d.CONSTANT
        elif basis.lower() in ['linear']:
            _basis = basisfunctionbem3d.LINEAR
        else:
            raise TypeError

        bem = new_slp_helmholtz_bem3d(k, mesh.surface3d, q_reg, q_sing, _basis,
                                      _basis)
        root = build_bem3d_cluster(bem, clf, _basis)

        if strict:
            broot = build_strict_block(root, root, eta, admis)
        else:
            broot = build_nonstrict_block(root, root, eta, admis)

        if aprx.lower() in ['aca']:
            setup_hmatrix_aprx_inter_row_bem3d(bem, root, root, broot, m)
        elif aprx.lower() in ['paca']:
            setup_hmatrix_aprx_paca_bem3d(bem, root, root, broot, eps_aca)
        elif aprx.lower() in ['hca']:
            setup_hmatrix_aprx_hca_bem3d(bem, root, root, broot, m, eps_aca)
        elif aprx.lower() in ['inter_row']:
            setup_hmatrix_aprx_inter_row_bem3d(bem, root, root, broot, m)

        Z = build_from_block_hmatrix(broot, rk)
        start = timer()
        assemble_bem3d_hmatrix(bem, broot, Z)
        time_assemble = timer() - start

        self._mat = Z
        self._time_assemble = time_assemble
        # keep references to h2lib objects so they don't get garbage collected
        self._root = root
        # important! don't ref bem and broot otherwise processes fail to terminate (not sure why)
        # self._bem = bem
        self._broot = broot

    def __del__(self):
        del self._mat
        del self._root
        # del self._bem
        del self._broot

    @property
    def time_assemble(self):
        return self._time_assemble

    __repr__ = _z_repr
