
import numpy as np
from cmut_nonlinear_sim.mesh import square
import argparse


def main():
    
    # define and parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('file')
    parser.add_argument('xl', type=float)
    parser.add_argument('yl', type=float)
    parser.add_argument('refn', nargs=2)
    args = vars(parser.parse_args())

    file = args['file']
    xl = args['xl']
    yl = args['yl']
    r_start, r_stop = args['refn']

    refns = range(r_start, r_stop)
    verts = []
    for refn in refns:
        mesh = square(xl, yl, refn=refn)
        verts.append(mesh.vertices)

    np.savez(file, refns=refns, verts=verts)

if __name__ == '__main__':
    main()