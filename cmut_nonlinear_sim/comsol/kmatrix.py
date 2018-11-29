'''
'''
import numpy as np
from numpy import linalg
import os
import matlab
import matlab.engine

_mateng = matlab.engine.start_matlab()
path = os.path.dirname(os.path.abspath(__file__))
_mateng.cd(str(os.path.normpath(path)), nargout=0)

import atexit
atexit.register(_mateng.quit)


def connect_comsol(ip='localhost', port=2036, path=None):
    ''''''
    if path is None:
        if os.name == 'nt':
            path = 'C:\Program Files\COMSOL\COMSOL53a\Multiphysics\mli'
        else:
            path = '//system//software//generic//comsol//5.3//mli'
    _mateng.addpath(path, nargout=0)
    _mateng.mphstart(ip, float(port), nargout=0) # float (not int) because MATLAB is dumb


def _mat_to_ndarray(mat):
    return np.array(mat).squeeze()


def _ndarray_to_mat(array, orient='row'):
    sz = None
    if array.ndim == 1:
        sz = 1, array.size if orient.lower() in ('row',) else array.size, 1
    return matlab.double(initializer=array.tolist(), size=sz)


def square_membrane(verts, lx, ly, lz, rho, ymod, pratio, fine=2, dx=None):
    ''''''
    if dx is None: 
        dx = lx / 10
    verts = _ndarray_to_mat(verts.T)
    ret = _mateng.comsol_square_membrane(verts, lx, ly, lz, rho, ymod, pratio, fine, dx)
    return linalg.inv(_mat_to_ndarray(ret[0]))


def square_membrane_from_mesh(lx, ly, lz, rho, ymod, pratio, fine=2, dx=None, refn=2):
    ''''''
    from . import mesh
    sq = mesh.square(lx, ly, refn=refn)
    if dx is None:
        dx = sq.hmin / 2
    return square_membrane(sq.vertices, lx, ly, lz, rho, ymod, pratio, fine, dx)


def circle_membrane():
    raise NotImplementedError


if __name__ == '__main__':

    file = None
    verts = np.load('scratch/square.npy')
    lx = 40e-6
    ly = 40e-6
    lz = 2e-6
    rho = 2200
    ymod = 70e9
    pratio = 0.17
    fine = 2
    dx = lx / 10

    connect_comsol()
    k = square_membrane(verts, lx, ly, lz, rho, ymod, pratio, fine, dx)

