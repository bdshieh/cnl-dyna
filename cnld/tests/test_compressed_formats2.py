
import pytest
import numpy as np
import scipy.sparse as sps

from cnld.compressed_formats2 import *
from cnld.h2lib import *
from cnld.mesh import fast_matrix_array


@pytest.fixture
def make_npfull():
    def f(shape=(20,20)):
        return np.ones(shape)
    return f

@pytest.fixture
def make_npsparse():
    def f(shape=(20,20)):
        return sps.csr_matrix(np.ones(shape))
    return f

@pytest.fixture
def make_full(make_npfull):
    def f(shape=(20,20)):
        return FullFormat(AMatrix.from_array(make_npfull(shape)))
    return f

@pytest.fixture
def make_sparse(make_npsparse):
    def f(shape=(20,20)):
        return SparseFormat(SparseMatrix.from_array(make_npsparse(shape)))
    return f

@pytest.fixture
def make_h():
    def f():
        mesh = fast_matrix_array(5, 5, 60e-6, 60e-6, 40e-6, 40e-6, refn=4)
        k  = 2 * np.pi * 1e6 / 1500.
        basis = basisfunctionbem3d.LINEAR
        m = 4
        q_reg = 2
        q_sing = 4
        admis = '2'
        eta = 1.1
        eps = 1e-12
        eps_aca = 1e-2
        clf = 16
        rk = 0
        
        bem = new_slp_helmholtz_bem3d(k, mesh.surface3d, q_reg, q_sing, basis, basis)
        root = build_bem3d_cluster(bem, clf, basis)
        broot = build_strict_block(root, root, eta, admis)
        setup_hmatrix_aprx_paca_bem3d(bem, root, root, broot, eps_aca)
        H = build_from_block_hmatrix(broot, rk)
        identity_hmatrix(H)

        return HFormat(H)
    return f


class TestFullFormat:

    def test_init(self, make_full):
        mat = make_full()
        assert mat.shape == (20, 20)
        assert mat.size > 0
    
    def test_add_full(self, make_full):
        mat = make_full() + make_full()
        assert mat[0, 0] == 2

    def test_add_sparse(self, make_full, make_sparse):
        mat = make_full() + make_sparse()
        assert mat[0, 0] == 2

    def test_add_h(self, make_full, make_h):
        H = make_h()
        F = make_full(H.shape)
        mat = F + H
        assert mat[0, 0] == 2

    def test_dim_mismatch(self, make_full, make_h):
        with pytest.raises(ValueError):
            mat = make_full() + make_h()

    def test_mul(self, make_full):
        A = make_full()

        mat = A * 5
        assert mat[0, 0] == 5

        mat = 4 * A
        assert mat[0, 0] == 4

        mat = A * np.ones(20)
        assert mat.shape == (20,)

        mat = A * make_full()
        assert mat.shape == (20, 20)
    
    def test_transpose(self):
        pass
    
    def test_adjoint(self):
        pass
    
    def test_lu(self):
        pass
    
    def test_chol(self):
        pass



class TestHFormat:

    def test_init(self, make_h):
        mat = make_h()
        assert mat.size > 0
    
    def test_add(self, make_h, make_sparse, make_full):
        H = make_h()

        mat = H + make_full(H.shape)

        mat = H + make_sparse(H.shape)

        mat = H + H

    def test_mul(self, make_h, make_full):
        H = make_h()

        mat = H * 5

        mat = 4 * H

        mat = H * np.ones(H.shape[1])
        assert mat.shape == (H.shape[0],)

        mat = H * H
    
    def test_transpose(self):
        pass
    
    def test_adjoint(self):
        pass
    
    def test_lu(self):
        pass
    
    def test_chol(self):
        pass


# class TestMbkFullMatrix:

#     def test_init(self, npfull):
#         MBK = MbkFullMatrix(npfull)
#         assert MBK.shape == (20, 20)
#         assert MBK.size > 0


# class TestMbkSparseMatrix:

#     def test_init(self, npsparse):
#         MBK = MbkSparseMatrix(npsparse)
#         assert MBK.shape == (20, 20)
#         assert MBK.size > 0
#         assert MBK.nz > 0


# class TestZFullMatrix:
    
#     def test_init(self):
#         pass
