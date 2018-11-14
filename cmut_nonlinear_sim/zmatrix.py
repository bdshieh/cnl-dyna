## zmatrix.py ##

from . h2lib import *
from timeit import default_timer as timer



class HierarchicalMatrix:

    _hmatrix = None

    def __init__(self, mesh, k, basis='linear', m=4, q_reg=2, q_sing=4,
         aprx='paca', admis='2', eta=1.4, eps=1e-12, eps_aca=1e-2, clf=None, rk=None):

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
        del self._bem, self._root, self._broot, self._hmatrix

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
        obj._root = None
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
        y = AVector(x.dim)
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

        return HierarchicalMatrix.from_hmatrix(Z_chol)

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

        return HierarchicalMatrix.from_hmatrix(Z_lu)
    
    def lusolve(self, b):

        Z_lu = self._hmatrix
        x = AVector.from_array(b)

        lrsolve_hmatrix_avector(False, Z_lu, x)

        return np.asarray(x.v)


class DenseMatrix:

    _amatrix = None

    def __init__(self, mesh, k, basis='linear', m=4, q_reg=2, q_sing=4):

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
        del self._bem, self._amatrix

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

        return DenseMatrix.from_amatrix(Z_chol)

    def cholsolve(self, b):
        
        Z_chol = self._amatrix
        x = AVector.from_array(b)

        cholsolve_amatrix_avector(Z_chol, x)

        return np.asarray(x.v)

    def lu(self, eps=1e-12):
        
        Z = self._amatrix
        Z_lu = clone_amatrix(Z)

        lrdecomp_amatrix(Z_lu)

        return DenseMatrix.from_amatrix(Z_lu)
    
    def lusolve(self, b):

        Z_lu = self._amatrix
        x = AVector.from_array(b)

        lrsolve_amatrix_avector(False, Z_lu, x)

        return np.asarray(x.v)


def lu(Z, eps=1e-12):
    return Z.lu(eps)


def chol(Z, eps=1e-12):
    return Z.chol(eps)


def lusolve(Z, b):
    return Z.lusolve(b)


def cholsolve(Z, b):
    return Z.cholsolve(b)





class ZMatrix:

    def __init__(self):
        pass
    

    @property
    def shape(self):
        pass
    
    @property
    def size(self):
        pass
    
    def dot(self, other):
        pass
    
    def cholesky(self):
        pass
    
    def lu(self):
        pass
    



