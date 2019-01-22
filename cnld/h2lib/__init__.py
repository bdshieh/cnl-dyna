
from . basic import *
from . avector import *
from . amatrix import *
from . cluster import *
from . block import *
from . surface3d import *
from . macrosurface3d import *
from . bem3d import *
from . rkmatrix import *
from . hmatrix import *
from . helmholtzbem3d import *
from . truncation import *
from . krylovsolvers import *
from . harith import *
from . factorizations import *
from . sparsematrix import *

init_h2lib()
# print('h2lib initialized')

import atexit

def uninit():
    uninit_h2lib()
    # print('h2lib uninitialized')

atexit.register(uninit)