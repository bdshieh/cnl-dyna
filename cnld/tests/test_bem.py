'''
'''
import pytest
import numpy as np
import scipy.sparse as sps
import os 

from cnld import abstract, bem
from cnld.mesh import Mesh
import cnld.arrays.matrix


test_dir = os.path.dirname(os.path.realpath(__file__))

''' FIXTURES '''

@pytest.fixture
def array():
    Config = cnld.arrays.matrix.Config
    cfg = Config()
    cfg.nelem = 3, 3
    cfg.kmat_file = os.path.join(test_dir, 'kmat.npz')
    return cnld.arrays.matrix.main(cfg, None)

@pytest.fixture
def mesh(array):
    return Mesh.from_abstract(array, refn=3)
    
@pytest.fixture
def hmargs():
    hmargs = {}
    hmargs['basis'] = 'linear'
    hmargs['m'] = 4
    hmargs['q_reg'] = 2
    hmargs['q_sing'] = 4
    hmargs['admis'] = '2'
    hmargs['eta'] = 1.1
    hmargs['eps'] = 1e-12
    hmargs['eps_aca'] = 1e-2
    hmargs['clf'] = 16
    hmargs['rk'] = 0
    return hmargs


''' TESTS '''

@pytest.mark.filterwarnings('ignore::PendingDeprecationWarning')
def test_mbk_from_abstract(array):
    MBK = bem.mbk_from_abstract(array, f=1e6, refn=3, format='SparseFormat')
    assert MBK.size > 0

    MBK = bem.mbk_from_abstract(array, f=1e6, refn=3, format='FullFormat')
    assert MBK.size > 0

def test_z_from_mesh(mesh, hmargs):
    k = 2 * np.pi * 1e6 / 1500.

    Z = bem.z_from_mesh(mesh, k, format='HFormat', **hmargs)
    assert Z.size > 0

    Z = bem.z_from_mesh(mesh, k, format='FullFormat', **hmargs)
    assert Z.size > 0

def test_z_from_abstract(array, hmargs):
    k = 2 * np.pi * 1e6 / 1500.

    Z = bem.z_from_abstract(array, k, refn=3, format='HFormat', **hmargs)
    assert Z.size > 0

    Z = bem.z_from_abstract(array, k, refn=3, format='FullFormat', **hmargs)
    assert Z.size > 0



