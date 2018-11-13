

from cmut_nonlinear_sim.mesh import *
import numpy as np
from matplotlib import pyplot as plt
import sys


if __name__ == '__main__':

    if len(sys.argv[1:]):
        refn = int(sys.argv[1])
    else:
        refn = 2

    s = square(2, 2, refn=refn)
    s.draw()

    c = circle(2, refn=refn)
    c.draw()

    ma = matrix_array(5, 5, 60e-6, 60e-6, refn=3, lengthx=40e-6, lengthy=40e-6)
    ma.draw()

    ma = matrix_array(5, 5, 60e-6, 60e-6, refn=3, shape='circle', radius=20e-6)
    ma.draw()



