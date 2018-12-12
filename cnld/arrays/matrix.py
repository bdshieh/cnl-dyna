''' 
Abstract representation of a matrix array.
'''
import numpy as np

from cnld.abstract import *
from cnld import util





def main(cfg, args):

    nmem_x, nmem_y = cfg.nmem
    mempitch_x, mempitch_y = cfg.mempitch
    length_x, length_y = cfg.length
    electrode_x, electrode_y = cfg.electrode
    npatch_x, npatch_y = cfg.npatch
    nelem_x, nelem_y = cfg.nelem
    elempitch_x, elempitch_y = cfg.elempitch

    # membrane properties
    memprops = {}
    memprops['length_x'] = length_x
    memprops['length_y'] = length_y
    memprops['electrode_x'] = electrode_x
    memprops['electrode_y'] = electrode_y
    memprops['y_modulus'] = cfg.y_modulus
    memprops['p_ratio'] = cfg.p_ratio
    memprops['isolation'] = cfg.isolation
    memprops['permittivity'] = cfg.permittivity
    memprops['gap'] = cfg.gap
    memprops['npatch_x'] = npatch_x
    memprops['npatch_y'] = npatch_y
    memprops['thickness'] = cfg.thickness
    memprops['density'] = cfg.density
    memprops['att'] = cfg.att
    memprops['kmat_file'] = cfg.kmat_file

    # calculate patch positions
    patchpitch_x = length_x / npatch_x
    patchpitch_y = length_y / npatch_y
    xx, yy, zz = np.meshgrid(np.linspace(0, (npatch_x - 1) * patchpitch_x, npatch_x),
                            np.linspace(0, (npatch_y - 1) * patchpitch_y, npatch_y),
                            0)
    patch_pos = np.c_[xx.ravel(), yy.ravel(), zz.ravel()] - [(npatch_x - 1) * patchpitch_x / 2,
                                                        (npatch_y - 1) * patchpitch_y / 2,
                                                        0]

    # calculate membrane positions
    xx, yy, zz = np.meshgrid(np.linspace(0, (nmem_x - 1) * mempitch_x, nmem_x),
                             np.linspace(0, (nmem_y - 1) * mempitch_y, nmem_y),
                             0)
    mem_pos = np.c_[xx.ravel(), yy.ravel(), zz.ravel()] - [(nmem_x - 1) * mempitch_x / 2,
                                                           (nmem_y - 1) * mempitch_y / 2,
                                                           0]

    # calculate element positions
    xx, yy, zz = np.meshgrid(np.linspace(0, (nelem_x - 1) * elempitch_x, nelem_x),
                             np.linspace(0, (nelem_y - 1) * elempitch_y, nelem_y),
                             0)
    elem_pos = np.c_[xx.ravel(), yy.ravel(), zz.ravel()] - [(nelem_x - 1) * elempitch_x / 2,
                                                           (nelem_y - 1) * elempitch_y / 2,
                                                           0]

    # construct element list
    elements = []
    elem_counter = 0
    mem_counter = 0
    patch_counter = 0

    for epos in elem_pos:
        # construct membrane list
        membranes = []
        for mpos in mem_pos:
            # construct patch list
            patches = []
            for ppos in patch_pos:

                # construct patch
                p = Patch()
                p.id = patch_counter
                p.position = (epos + mpos + ppos).tolist()
                p.length_x = patchpitch_x
                p.length_y = patchpitch_y

                patches.append(p)
                patch_counter += 1

            # construct membrane
            m = SquareCmutMembrane(**memprops)
            m.id = mem_counter
            m.position = (epos + mpos).tolist()
            m.patches = patches

            membranes.append(m)
            mem_counter += 1

        # construct element
        elem = Element()
        elem.id = elem_counter
        elem.position = epos.tolist()
        elem.kind = 'both'
        elem.membranes = membranes
        # element_position_from_membranes(elem)
        elements.append(elem)
        elem_counter += 1

    # construct array
    array = Array()
    array.id = 0
    array.elements = elements
    array.position = [0, 0, 0]

    return array


# default configuration
Config = {}
# membrane properties
Config['length'] = [40e-6, 40e-6]
Config['electrode'] = [40e-6, 40e-6]
Config['thickness'] = [2e-6,]
Config['density'] = [2040,]
Config['y_modulus'] = [110e9,]
Config['p_ratio'] = [0.22,]
Config['isolation'] = 200e-9
Config['permittivity'] = 6.3
Config['gap'] = 100e-9
Config['att'] = 0
Config['npatch'] = [3, 3]
Config['kmat_file'] = None
# array properties
Config['mempitch'] = [60e-6, 60e-6]
Config['nmem'] = [1, 1]
Config['elempitch'] = [60e-6, 60e-6]
Config['nelem'] = [5, 5]

if __name__ == '__main__':

    import sys
    from cnld import util

    # get script parser and parse arguments
    parser = util.script_parser(main, Config)
    args = parser.parse_args()
    array = args.func(args)

    if array is not None and args.file:
        dump(array, args.file)
    
