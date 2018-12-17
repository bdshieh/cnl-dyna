
import numpy as np
from cnld.mesh import square
import argparse


def main(cfg, args):
    
    r_start, r_stop = cfg.refns

    refns = range(r_start, r_stop)
    verts = []
    on_bounds = []
    for refn in refns:
        mesh = square(cfg.xl, cfg.yl, refn=refn)
        verts.append(mesh.vertices)
        on_bounds.append(mesh.on_boundary)

    np.savez(args.file, refns=refns, verts=verts, on_bounds=on_bounds)


if __name__ == '__main__':

    import sys
    from cnld import util

    # define configuration for this script
    Config = {}
    Config['xl'] = 40e-6
    Config['yl'] = 40e-6
    Config['refns'] = 2, 10

    # get script parser and parse arguments
    parser, run_parser = util.script_parser(main, Config)
    args = parser.parse_args()
    args.func(args)