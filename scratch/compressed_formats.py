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


class HFormat:

    _hmatrix = None

    def __init__(self, mesh, k, basis='linear', m=4, q_reg=2, q_sing=4,
         aprx='paca', admis='2', eta=1.4, eps=1e-12, eps_aca=1e-2, strict=True, 
         clf=None, rk=None, **kwargs):

        if rk is None:
            rk = m * m * m
        
        if clf is None:
            clf = 2 * m * m * m

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
        stop = timer() - start

        size = getsize_hmatrix(Z) / 1024. / 1024.
        shape = getrows_hmatrix(Z), getcols_hmatrix(Z)

        self._hmatrix = Z
        self._size = size
        self._shape = shape
        self._assemble_time = stop
        self._bem = bem
        self._root = root
        self._broot = broot

    def __del__(self):
        if self._bem is not None: del self._bem
        if self._root is not None: del self._root
        if self._broot is not None: del self._broot
        if self._hmatrix is not None: del self._hmatrix

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
    def size(self):
        return self._size
    
    @property
    def shape(self):
        return self._shape

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
        raise NotImplementedError

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


class FullFormat:

    _amatrix = None

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
        stop = timer() - start

        size = getsize_amatrix(Z) / 1024. / 1024.
        shape = Z.rows, Z.cols

        self._amatrix = Z
        self._size = size
        self._shape = shape
        self._assemble_time = stop
        self._bem = bem

    def __del__(self):
        if self._bem is not None: del self._bem
        if self._amatrix is not None: del self._amatrix

    @classmethod
    def from_amatrix(cls, a):
        
        obj = cls.__new__(cls)

        size = getsize_amatrix(a) / 1024. / 1024.
        shape = a.rows, a.cols

        obj._amatrix = a
        obj._size = size
        obj._shape = shape
        obj._assemble_time = None
        obj._bem = None
        
        return obj

    @property
    def size(self):
        return self._size
    
    @property
    def shape(self):
        return self._shape

    @property
    def assemble_time(self):
        return self._assemble_time

    def _matvec(self, v):
        
        Z = self._amatrix
        x = AVector.from_array(v)
        y = AVector(x.dim)
        clear_avector(y)
        
        addeval_amatrix_avector(1.0, Z, x, y)

        return np.asarray(y.v)

    def _matmat(self, m):
        raise NotImplementedError

    def dot(self, v):

        if v.squeeze().ndim == 1:
            return self._matvec(v)
        else:
            return self._matmat(v)

    def chol(self, eps=1e-12):
        
        Z = self._amatrix
        Z_chol = clone_amatrix(Z)
        choldecomp_amatrix(Z_chol)
        return FullFormat.from_amatrix(Z_chol)

    def cholsolve(self, b):
        
        Z_chol = self._amatrix
        x = AVector.from_array(b)
        cholsolve_amatrix_avector(Z_chol, x)
        return np.asarray(x.v)

    def lu(self, eps=1e-12):
        
        Z = self._amatrix
        Z_lu = clone_amatrix(Z)
        lrdecomp_amatrix(Z_lu)
        return FullFormat.from_amatrix(Z_lu)
    
    def lusolve(self, b):

        Z_lu = self._amatrix
        x = AVector.from_array(b)
        lrsolve_amatrix_avector(False, Z_lu, x)
        return np.asarray(x.v)

    def draw(self):
        pass


class SparseFormat:

    _sparsematrix = None

    def __init__(self, a):

        start = timer()
        A = SparseMatrix.from_array(a)
        stop = timer() - start

        size = getsize_sparsematrix(A) / 1024. / 1024.
        shape = A.rows, A.cols

        self._sparsematrix = A
        self._size = size
        self._shape = shape
        self._assemble_time = stop

    def __del__(self):
        if self._sparsematrix is not None: del self._sparsematrix

    @property
    def size(self):
        return self._size
    
    @property
    def shape(self):
        return self._shape

    @property
    def assemble_time(self):
        return self._assemble_time
    
    @property
    def nnz(self):
        return self._sparsematrix.nz

    def to_hformat(self, href):

        hm = clonestructure_hmatrix(href)
        copy_sparsematrix_hmatrix(self._sparsematrix, hm)
        return hm
