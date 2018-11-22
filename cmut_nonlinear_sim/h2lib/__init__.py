
from . basic_cy import *
from . avector_cy import *
from . amatrix_cy import *
from . cluster_cy import *
from . block_cy import *
from . surface3d_cy import *
from . macrosurface3d_cy import *
from . bem3d_cy import *
from . rkmatrix_cy import *
from . hmatrix_cy import *
from . helmholtzbem3d_cy import *
from . truncation_cy import *
from . krylovsolvers_cy import *
from . harith_cy import *
from . factorizations_cy import *
from . sparsematrix_cy import *

init_h2lib()
# print('h2lib initialized')

import atexit
atexit.register(uninit_h2lib)