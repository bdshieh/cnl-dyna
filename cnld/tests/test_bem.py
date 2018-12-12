'''
'''
import pytest
import numpy as np
import scipy.sparse as sps

from cnld import arrays, abstract, bem


''' FIXTURES '''

@pytest.fixture
def array():
    array = arrays.matrix()


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
    
    def test_lu(self, make_full):
        mat = make_full()
        LU = mat.lu()
    
    def test_chol(self, make_full):
        mat = make_full()
        CHOL = mat.chol()



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
    
    def test_lu(self, make_h):
        mat = make_h()
        LU = mat.lu()
    
    def test_chol(self, make_h):
        mat = make_h()
        CHOL = mat.chol()


# class TestMbkFullMatrix:

#     def test_init(self, npfull):
#         MBK = MbkFullMatrix(npfull)
#         assert MBK.shape == (20, 20)
#         assert MBK.size > 0


class TestMbkSparseMatrix:

    def test_init(self, make_npsparse):
        MBK = MbkSparseMatrix(make_npsparse())
        assert MBK.shape == (20, 20)
        assert MBK.size > 0
        assert MBK.nnz > 0


class TestZFull:
    
    def test_init(self):
        pass


class TestZHMatrix:
    
    def test_init(self):

        mesh = fast_matrix_array(4, 4, 60e-6, 60e-6, 40e-6, 40e-6, refn=4)
        k  = 2 * np.pi * 1e6 / 1500.
        
        m = 4
        q_reg = 2
        q_sing = 4
        admis = '2'
        eta = 1.1
        eps = 1e-12
        eps_aca = 1e-2
        strict = True
        clf = 16
        rk = 0

        nvert = len(mesh.vertices)
        ntri = len(mesh.triangles)

        basis = 'constant'
        aprx = 'paca'
        Z = ZHMatrix(mesh, k, basis, m, q_reg, q_sing, aprx, admis, eta, eps, eps_aca, strict, clf, rk)
        assert Z.size > 0
        assert Z.shape == (ntri, ntri)

        basis = 'linear'
        aprx = 'aca'
        Z = ZHMatrix(mesh, k, basis, m, q_reg, q_sing, aprx, admis, eta, eps, eps_aca, strict, clf, rk)
        assert Z.size > 0
        assert Z.shape == (nvert, nvert)

        aprx = 'paca'
        Z = ZHMatrix(mesh, k, basis, m, q_reg, q_sing, aprx, admis, eta, eps, eps_aca, strict, clf, rk)
        assert Z.size > 0
        assert Z.shape == (nvert, nvert)
    
        aprx = 'hca'
        Z = ZHMatrix(mesh, k, basis, m, q_reg, q_sing, aprx, admis, eta, eps, eps_aca, strict, clf, rk)
        assert Z.size > 0
        assert Z.shape == (nvert, nvert)

        aprx = 'inter_row'
        Z = ZHMatrix(mesh, k, basis, m, q_reg, q_sing, aprx, admis, eta, eps, eps_aca, strict, clf, rk)
        assert Z.size > 0
        assert Z.shape == (nvert, nvert)