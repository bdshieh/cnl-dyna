## zmatrix.py ##

from . h2lib import *
from timeit import default_timer as timer



class HierarchicalMatrix:

    _hmatrix = None

    def __init__(self, mesh, k, basis='linear', q_reg=2, q_sing=4,
        clf=128, admis='2', eta=1.4, aprx='paca', m=4, rk=64, eps=1e-12):

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
            setup_hmatrix_aprx_paca_bem3d(bem, root, root, broot, eps)
        elif aprx.lower() in  ['hca']:
            setup_hmatrix_aprx_hca_bem3d(bem, root, root, broot, m, eps)

        Z = build_from_block_hmatrix(broot, m * m * m)

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


        # del bem, root, broot
    
    @classmethod
    def from_hmatrix(cls, hm):
        
        obj = cls.__new__(cls)

        size = getsize_hmatrix(Z) / 1024. / 1024.
        shape = getrows_hmatrix(Z), getcols_hmatrix(Z)

        obj._size = size
        obj._shape = shape
        obj._assemble_time = None
        
        return obj

    @property
    def size(self):
        return self._size
    
    @property
    def shape(self):
        return self._shape

    def _matvec(self, v):
        
        Z = self._hmatrix
        # x = AVector.from_array(v)
        x = AVector(v.size)
        random_avector(x)
        y = AVector(x.dim)
        clear_avector(y)
        
        addeval_hmatrix_avector(1.0, Z, x, y)
        # addevalsymm_hmatrix_avector(1.0, Z, x, y)

        return np.asarray(y.v)

    def _choldecomp(self, eps):
        
        Z = self._hmatrix
        Z_chol = clone_hmatrix(Z)
        tm = new_releucl_truncmode()

        choldecomp_hmatrix(Z_chol, tm, eps)

        del tm

        return HierarchicalMatrix.from_hmatrix(Z_chol)
    


class DenseMatrix:
    pass





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
    



