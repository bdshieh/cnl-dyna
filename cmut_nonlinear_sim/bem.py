## bem.py ##

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt

from . compressed_formats import ZMatrix, MBKMatrix
from . import h2lib
from . mesh import from_spec



def calc_transfer_funcs():
    pass


def function(spec):
    pass

    # for frequency each f, create mesh from spec
    # from mesh, generate M, B, K as Scipy sparse matrices
    # convert M, B, K to MBK in compressed sparseformat
    # generate Z in compressed hformat
    # perform G = MBK + Z by converting MBK to hformat and adding
    # decompose LU = G
    # solve x(f) using LU for lumped forcings
    # calculate time impulse response using ifft

