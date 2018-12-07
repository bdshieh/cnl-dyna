## zmatrix.py ##

from . h2lib import *
from timeit import default_timer as timer

from matplotlib import pyplot as plt
from matplotlib import patches


class MBKMatrix(self):

    _matrix = None

    def __init__(self, a):
        self._matrix = SparseFormat(a)

    def __repr__(self):

        repr = []
        repr.append('MBKMatrix (Mass, Damping, Stiffness Matrix)\n')
        repr.append(f'  Format: {self.format}\n')
        repr.append(f'  Shape: {self.shape}\n')
        repr.append(f'  Size: {self.size:.2f} MB\n')
        return ''.join(repr)

    def __str__(self):
        return self.__repr__

    @property
    def format(self):
        return self._matrix.__class__.__name__

    @property
    def shape(self):
        return self._matrix.shape

    @property
    def size(self):
        return self._matrix.size

    @property
    def assemble_time(self):
        return self._matrix._assemble_time

    def as_hformat(self, href):
        
        sm = self._matrix
        assert isinstance(sm, SparseFormat)

        self._matrix = sm.to_hformat(href)
        del sm

    def add(self, other, eps=1e-12):

        sm = self._matrix
        om = other._matrix
        assert isinstance(sm, HFormat)
        assert isinstance(om, HFormat)

        tm = new_releucl_truncmode()
        add_hmatrix(1.0, om, tm, eps, sm)


class ZMatrix:

    _matrix = None

    def __init__(self, format, mesh, k, *args, **kwargs):
        
        if format.lower() in ['f', 'full', 'fullmatrix', 'fullformat']:
            self._matrix = FullFormat(mesh, k, *args, **kwargs)       
        elif format.lower() in ['h', 'hmat', 'hmatrix', 'hformat']:
            self._matrix = HFormat(mesh, k, *args, **kwargs)
        else:
            raise TypeError

    def __repr__(self):

        repr = []
        repr.append('ZMatrix (Acoustic Impedance Matrix)\n')
        repr.append(f'  Format: {self.format}\n')
        repr.append(f'  Shape: {self.shape}\n')
        repr.append(f'  Size: {self.size:.2f} MB\n')
        return ''.join(repr)
    
    def __str__(self):
        return self.__repr__
    
    @property
    def format(self):
        return self._matrix.__class__.__name__

    @property
    def shape(self):
        return self._matrix.shape
    
    @property
    def size(self):
        return self._matrix.size

    @property
    def assemble_time(self):
        return self._matrix._assemble_time

    def dot(self, other):
        return self._matrix.dot(other)
    
    def cholesky(self, eps=1e-12):
        return self._matrix.cholesky(eps)
    
    def lu(self, eps=1e-12):
        return self._matrix.lu(eps)
    
    def lusolve(self, b):
        return self._matrix.lusolve(b)

    def cholsolve(self, b):
        return self._matrix.cholsolve(b)

    def draw(self, *args, **kwargs):
        self._matrix.draw(*args, **kwargs)


def lu(Z, eps=1e-12):
    return Z.lu(eps)

def chol(Z, eps=1e-12):
    return Z.chol(eps)

def lusolve(Z, b):
    return Z.lusolve(b)

def cholsolve(Z, b):
    return Z.cholsolve(b)

class Format:

    def __init__(self, mat):
        self._mat = mat

    def __del__(self):
        if self._mat: del self._mat

    @property
    def size(self):
        return self._mat._size
    
    @property
    def shape(self):
        return self._mat._shape

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

    def __add__(self, x):

        if isinstance(x, FullFormat):
            add_amatrix(1.0, False, x._mat, self._mat)
        elif isinstance(x, SparseFormat):
            add_sparsematrix_amatrix(1, False, x._mat, self._mat)
        elif isinstance(x, HFormat):
            add_hmatrix_amatrix(1, False, x._mat, self._mat)
        else:
            raise ValueError('operation with unrecognized type')
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


class SparseFormat:

    @property
    def nnz(self):
        return self._mat.nz

    def __add__(self, x):

        if isinstance(x, FullFormat):
            raise NotImplementedError('operation not supported with this type')
        elif isinstance(x, SparseFormat):
            raise NotImplementedError('operation not supported with this type')
        elif isinstance(x, HFormat):
            raise NotImplementedError('operation not supported with this type')
        else:
            raise ValueError('operation with unrecognized type')
        return self

    def _smul(self, x):

        raise NotImplementedError('operation not supported with this type')
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
        
    def to_hformat(self, href):

        hm = clonestructure_hmatrix(href)
        copy_sparsematrix_hmatrix(self._sparsematrix, hm)
        return hm



class HFormat(Format):



    @classmethod
    def from_hmatrix(cls, hm):
        
        obj = cls.__new__(cls)

        size = getsize_hmatrix(hm) / 1024. / 1024.
        shape = getrows_hmatrix(hm), getcols_hmatrix(hm)

        obj._hmatrix = hm
        obj._size = size
        obj._shape = shape
        obj._assemble_time = None
        obj._bem = None
        obj._root = hm.rc
        obj._broot = None

        return obj

    @property
    def assemble_time(self):
        return self._assemble_time

    def _matvec(self, v):
        
        Z = self._hmatrix
        x = AVector.from_array(v)
        y = AVector(v.size)
        clear_avector(y)
        
        # addeval_hmatrix_avector(1.0, Z, x, y)
        addevalsymm_hmatrix_avector(1.0, Z, x, y)

        return np.asarray(y.v)

    def _matmat(self, m):
        return NotImplemented

    def dot(self, v):

        if v.squeeze().ndim == 1:
            return self._matvec(v)
        else:
            return self._matmat(v)

    def chol(self, eps=1e-12):
        
        Z = self._hmatrix
        Z_chol = clone_hmatrix(Z)
        tm = new_releucl_truncmode()
        choldecomp_hmatrix(Z_chol, tm, eps)
        del tm
        return HFormat.from_hmatrix(Z_chol)

    def cholsolve(self, b):
        
        Z_chol = self._hmatrix
        x = AVector.from_array(b)
        cholsolve_hmatrix_avector(Z_chol, x)
        return np.asarray(x.v)

    def lu(self, eps=1e-12):
        
        Z = self._hmatrix
        Z_lu = clone_hmatrix(Z)
        tm = new_releucl_truncmode()
        lrdecomp_hmatrix(Z_lu, tm, eps)
        del tm
        return HFormat.from_hmatrix(Z_lu)
    
    def lusolve(self, b):

        Z_lu = self._hmatrix
        x = AVector.from_array(b)
        lrsolve_hmatrix_avector(False, Z_lu, x)
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
        
        hm = self._hmatrix
        maxidx = len(hm.rc.idx), len(hm.cc.idx)

        fig, ax = plt.subplots(figsize=(9,9))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.invert_yaxis()
        ax.set_aspect('equal')
        self._draw_hmatrix(hm, (0, 0, 1, 1), maxidx, ax)
        fig.show()



        obj._shape = shape
        obj._assemble_time = None
        obj._bem = None
        
        return obj

    def _matvec(self, v):
        
        Z = self._amatrix
        x = AVector.from_array(v)
        y = AVector(x.dim)
        clear_avector(y)
        
        addeval_amatrix_avector(1.0, Z, x, y)

        return np.asarray(y.v)


    def _chol(self, eps=1e-12):
        
        Z = self._amatrix
        Z_chol = clone_amatrix(Z)
        choldecomp_amatrix(Z_chol)
        return FullFormat.from_amatrix(Z_chol)

    def _cholsolve(self, b):
        
        Z_chol = self._amatrix
        x = AVector.from_array(b)
        cholsolve_amatrix_avector(Z_chol, x)
        return np.asarray(x.v)

    def _lu(self, eps=1e-12):
        
        Z = self._amatrix
        Z_lu = clone_amatrix(Z)
        lrdecomp_amatrix(Z_lu)
        return FullFormat.from_amatrix(Z_lu)
    
    def _lusolve(self, b):

        Z_lu = self._amatrix
        x = AVector.from_array(b)
        lrsolve_amatrix_avector(False, Z_lu, x)
        return np.asarray(x.v)

    def draw(self):
        pass


