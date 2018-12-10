## zmatrix.py ##

from . h2lib import *
from timeit import default_timer as timer

from matplotlib import pyplot as plt
from matplotlib import patches
from scipy.sparse import csr_matrix, issparse




class Format:

    def __init__(self, mat):
        self._mat = mat

    def __del__(self):
        if self._mat is not None: del self._mat

    @property
    def shape(self):
        return self._mat.rows, self._mat.cols

    @property
    def format(self):
        return self.__class__.__name__
        
    def __add__(self, x):
        return NotImplemented

    def __mul__(self, x):
        return self.dot(x)

    def __call__(self, x):
        return self * other

    def __neg__(self):
        return self * -1

    def __sub__(self, x):
        return self.__add__(-x)

    def _smul(self, x):
        raise NotImplementedError

    def _matmat(self, x):
        return NotImplemented
    
    def _matvec(self, x):
        return NotImplemented

    def matmat(self, X):
        X = np.asanyarray(X)
        if X.ndim != 2:
            raise ValueError
        M, N = self.shape
        if X.shape[0] != N:
            raise ValueError

        Y = self._matmat(X)

        return Y
    
    def matvec(self, x):
        x = np.asanyarray(x)
        M, N = self.shape
        if x.shape != (N,) and x.shape != (N,1):
            raise ValueError('dimension mismatch')

        y = self._matvec(x)

        if x.ndim == 1:
            y = y.reshape(M)
        elif x.ndim == 2:
            y = y.reshape(M, 1)

        return y

    def dot(self, x):
        if isscalar(x):
            return self._smul(x)
        else:
            x = np.asarray(x)
            
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
    
    def _lu(self, eps):
        raise NotImplementedError
    
    def _chol(self, eps):
        raise NotImplementedError
    
    def _lusolve(self, b):
        raise NotImplementedError
    
    def _cholsolve(self, b):
        raise NotImplementedError

    def lu(self, eps=None):
        pass
    
    def lusolve(self, b):
        pass

    def chol(self, eps=None):
        pass
    
    def cholsolve(self, b):
        pass


class FullFormat(Format):

    @property
    def size(self):
        return getsize_amatrix(self._mat)

    def __add__(self, x):
        if isinstance(x, FullFormat):
            add_amatrix(1.0, False, x._mat, self._mat)
        elif isinstance(x, SparseFormat):
            add_sparsematrix_amatrix(1, False, x._mat, self._mat)
        elif isinstance(x, HFormat):
            add_hmatrix_amatrix(1, False, x._mat, self._mat)
        else:
            return NotImplemented
        return self

    def _smul(self, x):
        scale_amatrix(x, self._mat)
        return self

    def _matmat(self, x):
        if isinstance(x, FullFormat):
            addmul_amatrix(1.0, False, self._mat, x._mat)
        elif isinstance(x, SparseFormat):
            raise NotImplementedError('operation not supported with this type')
        elif isinstance(x, HFormat):
            raise NotImplementedError('operation not supported with this type')
        else:
            raise ValueError('operation with unrecognized type')
        return self

    def _matvec(self, x):
        x = AVector.from_array(x)
        y = AVector(x.size)
        clear_avector(y)
        addeval_hmatrix_avector(1.0, self._mat, x, y)
        # addevalsymm_hmatrix_avector(1.0, self._mat, x, y)
        return np.asarray(y.v)

    def _lu(self, eps):
        LU = clone_amatrix(self._mat)
        return FullFormat(lrdecomp_amatrix(LU))
    
    def _chol(self, eps):
        CH = clone_amatrix(self._mat)
        return FullFormat(choldecomp_amatrix(CH))
   
    def _lusolve(self, b):
        x = AVector.from_array(b)
        lrsolve_amatrix_avector(False, self._mat, x)
        return np.asarray(x.v)   

    def _cholsolve(self, b):
        x = AVector.from_array(b)
        cholsolve_amatrix_avector(self._mat, x)
        return np.asarray(x.v)


class SparseFormat(Format):

    @property
    def nnz(self):
        return self._mat.nz

    @property
    def size(self):
        return getsize_sparsematrix(self._mat)

    def __add__(self, x):
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
        return self

    def _matvec(self, x):
        x = AVector.from_array(x)
        y = AVector(x.size)
        clear_avector(y)
        addeval_sparsematrix_avector(1.0, self._mat, x, y)
        return np.asarray(y.v)

    def _lu(self, eps):
        raise NotImplementedError('operation not supported with this type')
    
    def _chol(self, eps):
        raise NotImplementedError('operation not supported with this type')
   
    def _lusolve(self, b):
        raise NotImplementedError('operation not supported with this type')  

    def _cholsolve(self, b):
        raise NotImplementedError('operation not supported with this type')
        
    def _as_hformat(self, href):
        hm = clonestructure_hmatrix(href)
        copy_sparsematrix_hmatrix(self._sparsematrix, hm)
        return HFormat(hm)


class HFormat(Format):

    eps_add = 1e-12
    eps_lu = 1e-12
    eps_chol = 1e-12

    @property
    def size(self):
        return getsize_hmatrix(self._mat)

    def __add__(self, x):
        if isinstance(x, FullFormat):
            tm = new_releucl_truncmode()
            add_amatrix_hmatrix(1.0, False, x._mat, tm, self.eps_add, self._mat)
        elif isinstance(x, SparseFormat):
            tm = new_releucl_truncmode()
            add_hmatrix(1, False, x._as_hformat(self._mat), self._mat)
        elif isinstance(x, HFormat):
            tm = new_releucl_truncmode()
            add_hmatrix(1, False, x._mat, tm, self.eps_add, self._mat)
        else:
            return NotImplemented
        return self

    def _smul(self, x):
        id = clonestructure_hmatrix(self._mat)
        identity_hmatrix(id)
        z = clonestructure_hmatrix(self._mat)
        clear_hmatrix(z)
        tm = new_releucl_truncmode()
        addmul_hmatrix(x, False, hm, False, self._mat, tm, self.eps_add, z)
        return HFormat(z)

    def _matmat(self, x):
        if isinstance(x, FullFormat):
            raise NotImplementedError('operation not supported with this type')     
        elif isinstance(x, SparseFormat):
            raise NotImplementedError('operation not supported with this type') 
        elif isinstance(x, HFormat):
            z = clonestructure_hmatrix(self._mat)
            clear_hmatrix(z)
            tm = new_releucl_truncmode()
            addmul_hmatrix(1.0, False, x, False, self._mat, tm, self.eps_add, z)
            return HFormat(z)
        else:
            raise ValueError('operation with unrecognized type')

    def _matvec(self, x):
        x = AVector.from_array(x)
        y = AVector(x.size)
        clear_avector(y)
        addeval_hmatrix_avector(1.0, self._mat, x, y)
        # addevalsymm_hmatrix_avector(1.0, self._mat, x, y)
        return np.asarray(y.v)

    def _lu(self):
        LU = clone_hmatrix(self._mat)
        tm = new_releucl_truncmode()
        return HFormat(lrdecomp_hmatrix(Z_lu, tm, self.eps_lu))

    def _chol(self):
        LU = clone_hmatrix(self._mat)
        tm = new_releucl_truncmode()
        return HFormat(choldecomp_hmatrix(Z_lu, tm, self.eps_chol))
   
    def _lusolve(self, b):
        x = AVector.from_array(b)
        lrsolve_hmatrix_avector(False, self._mat, x)
        return np.asarray(x.v)   

    def _cholsolve(self, b):
        x = AVector.from_array(b)
        cholsolve_hmatrix_avector(self._mat, x)
        return np.asarray(x.v)
    
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
            sq = patches.Rectangle((x0, y0), width, height, edgecolor='black', fill=fill, facecolor='black')
            ax.add_patch(sq)
            if rk: 
                fontsize = int(round((112 - 6) * width + 6))
                if width > 0.03: ax.text(x0 + 0.05 * width, y0 + 0.95 * height, rk, fontsize=fontsize)
        
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

        fig, ax = plt.subplots(figsize=(9,9))
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
    repr.append(f'  Format: {self.format}\n')
    repr.append(f'  Shape: {self.shape}\n')
    repr.append(f'  Size: {self.size:.2f} MB\n')
    return ''.join(repr)


def _z_repr(self):

    repr = []
    repr.append('ZMatrix (Acoustic Impedance Matrix)\n')
    repr.append(f'  Format: {self.format}\n')
    repr.append(f'  Shape: {self.shape}\n')
    repr.append(f'  Size: {self.size:.2f} MB\n')
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

    def __init__(self, mesh, k, basis='linear', m=4, q_reg=2, q_sing=4, **kwargs):

        if basis.lower() in ['constant']:
            _basis = basisfunctionbem3d.CONSTANT
        elif basis.lower() in ['linear']:
            _basis = basisfunctionbem3d.LINEAR
        else:
            raise TypeError

        bem = new_slp_helmholtz_bem3d(k, mesh.surface3d, q_reg, q_sing, _basis, _basis)

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

    def __init__(self, mesh, k, basis='linear', m=4, q_reg=2, q_sing=4,
         aprx='paca', admis='2', eta=1.4, eps=1e-12, eps_aca=1e-2, strict=True, 
         clf=None, rk=None, **kwargs):

        if basis.lower() in ['constant']:
            _basis = basisfunctionbem3d.CONSTANT
        elif basis.lower() in ['linear']:
            _basis = basisfunctionbem3d.LINEAR
        else:
            raise TypeError
        
        bem = new_slp_helmholtz_bem3d(k, mesh.surface3d, q_reg, q_sing, _basis, _basis)
        root = build_bem3d_cluster(bem, clf, _basis)

        if strict:
            broot = build_strict_block(root, root, eta, admis)
        else:
            broot = build_nonstrict_block(root, root, eta, admis)

        if aprx.lower() in ['aca']:
            setup_hmatrix_aprx_inter_row_bem3d(bem, root, root, broot, m)
        elif aprx.lower() in ['paca']:
            setup_hmatrix_aprx_paca_bem3d(bem, root, root, broot, eps_aca)
        elif aprx.lower() in  ['hca']:
            setup_hmatrix_aprx_hca_bem3d(bem, root, root, broot, m, eps_aca)
        elif aprx.lower() in ['inter_row']:
            setup_hmatrix_aprx_inter_row_bem3d(bem, root, root, broot, m)

        Z = build_from_block_hmatrix(broot, rk)
        start = timer()
        assemble_bem3d_hmatrix(bem, broot, Z)
        time_assemble = timer() - start

        self._mat = Z
        self._time_assemble = time_assemble
        # self._bem = bem
        # self._root = root
        # self._broot = broot

    @property
    def time_assemble(self):
        return self._time_assemble

    __repr__ = _z_repr