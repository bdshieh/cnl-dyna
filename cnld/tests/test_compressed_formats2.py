
import pytest
import numpy as np
import scipy.sparse as sps

from cnld.compressed_formats2 import *
from cnld.h2lib import *

@pytest.fixture
def make_npfull():
    def f():
        return np.ones((20, 20))
    return f

@pytest.fixture
def make_npsparse():
    def f():
        return sps.csr_matrix(np.ones((20, 20)))
    return f

@pytest.fixture
def make_full(make_npfull):
    def f():
        return FullFormat(AMatrix.from_array(make_npfull()))
    return f

@pytest.fixture
def make_sparse(make_npsparse):
    def f():
        return SparseFormat(SparseMatrix.from_array(make_npsparse()))
    return f

@pytest.fixture
def make_h():
    pass


class TestFullFormat:

    def test_init(self, make_full):
        mat = make_full()
        assert mat.shape == (20, 20)
        assert mat.size > 0
    
    def test_add_full(self, make_full):
        mat = make_full() + make_full()

    def test_add_sparse(self, make_full, make_sparse):
        mat = make_full() + make_sparse()

    def test_add_h(self, make_full, make_h):
        pass

    def test_mul(self):
        pass
    
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
