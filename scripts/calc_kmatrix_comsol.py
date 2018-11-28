'''
'''
import numpy as np
import argparse

from cmut_nonlinear_sim import abstract
from cmut_nonlinear_sim import comsol


def main(**kwargs):

    # define and parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('file')
    parser.add_argument('-r', '--refns', nargs=2, type=float)
    parser.add_argument('-o', '--overwrite', action='store_true')
    args = vars(parser.parse_args())

    file = args['file']
    spec = args['spec']
    fine = args['fine']
    r_start, r_stop = args['refn']
    array = abstract.load(spec)
    firstmem = array.elements[0].membranes[0]

    lx = firstmem.length_x
    ly = firstmem.length_y
    lz = firstmem.thickness[0]
    rho = firstmem.density[0]
    ymod = firstmem.ymodulus[0]
    pratio = firstmem.pratio[0]

    refn = range(r_start, r_stop)
    Ks = []
    for r in refn:
        K = comsol.square_membrane_from_mesh(lx, ly, lz, rho, ymod, pratio, fine, r)
        Ks.append(K)

    np.savez(file, K=Ks, refn=refn)


if __name__ == '__main__':
    main()