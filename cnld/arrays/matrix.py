''' 
Abstract representation of a matrix array.
'''
import numpy as np

from cnld.abstract import *
from cnld import util


def main(cfg, args):

    if cfg.shape.lower() in ['square', 's']:
        square = True
    elif cfg.shape.lower() in ['circle', 'circular', 'c']:
        square = False
    else:
        raise ValueError('Invalid shape property')

    nmem_x, nmem_y = cfg.nmem
    mempitch_x, mempitch_y = cfg.mempitch
    if square:
        length_x, length_y = cfg.length
        electrode_x, electrode_y = cfg.electrode
        npatch_x, npatch_y = cfg.npatch
    else:
        radius = cfg.radius
        electrode_r = cfg.electrode_r
        npatch_r, npatch_theta = cfg.npatch
    nelem_x, nelem_y = cfg.nelem
    elempitch_x, elempitch_y = cfg.elempitch

    # membrane properties
    memprops = {}
    if square:
        memprops['length_x'] = length_x
        memprops['length_y'] = length_y
        memprops['electrode_x'] = electrode_x
        memprops['electrode_y'] = electrode_y
    else:
        memprops['radius'] = radius
        memprops['electrode_r'] = electrode_r
    memprops['y_modulus'] = cfg.y_modulus
    memprops['p_ratio'] = cfg.p_ratio
    memprops['isolation'] = cfg.isolation
    memprops['permittivity'] = cfg.permittivity
    memprops['gap'] = cfg.gap
    memprops['damping_mode_a'] = cfg.damping_mode_a
    memprops['damping_mode_b'] = cfg.damping_mode_b
    memprops['damping_ratio_a'] = cfg.damping_ratio_a
    memprops['damping_ratio_b'] = cfg.damping_ratio_b
    memprops['thickness'] = cfg.thickness
    memprops['density'] = cfg.density

    # calculate patch positions
    if square:
        patchpitch_x = length_x / npatch_x
        patchpitch_y = length_y / npatch_y
        xx, yy, zz = np.meshgrid(np.linspace(0, (npatch_x - 1) * patchpitch_x, npatch_x),
                                np.linspace(0, (npatch_y - 1) * patchpitch_y, npatch_y),
                                0)
        patch_pos = np.c_[xx.ravel(), yy.ravel(), zz.ravel()] - [(npatch_x - 1) * patchpitch_x / 2,
                                                            (npatch_y - 1) * patchpitch_y / 2,
                                                            0]
    else:
        # patchpitch_r = radius / npatch_r
        # patchpitch_theta = 2 * np.pi / npatch_theta
        patch_r = np.linspace(0, radius, npatch_r + 1)
        patch_theta = np.linspace(0, 2 * np.pi, npatch_theta + 1)
        patch_rmin = [patch_r[i] for i in range(npatch_r) for j in range(npatch_theta)]
        patch_rmax = [patch_r[i + 1] for i in range(npatch_r) for j in range(npatch_theta)]
        patch_thetamin = [patch_theta[j] for i in range(npatch_r) for j in range(npatch_theta)]
        patch_thetamax = [patch_theta[j + 1] for i in range(npatch_r) for j in range(npatch_theta)]
        patch_pos = np.array([0, 0, 0])

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
            for i, ppos in enumerate(patch_pos):
                # construct patch
                p = Patch()
                p.id = patch_counter
                p.position = (epos + mpos + ppos).tolist()
                if square:
                    p.length_x = patchpitch_x
                    p.length_y = patchpitch_y
                else:
                    p.radius_min = patch_rmin[i]
                    p.radius_max = patch_rmax[i]
                    p.theta_min = patch_thetamin[i]
                    p.theta_max = patch_thetamax[i]

                patches.append(p)
                patch_counter += 1

            # construct membrane
            if square:
                m = SquareCmutMembrane(**memprops)
            else:
                m = CircularCmutMembrane(**memprops)
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
_Config = {}
# membrane properties
_Config['shape'] = 'square'
_Config['length'] = [40e-6, 40e-6]
_Config['electrode'] = [40e-6, 40e-6]
_Config['radius'] = 40e-6 / 2
_Config['electrode_r'] = 40e-6 / 2
_Config['thickness'] = [2e-6,]
_Config['density'] = [2040,]
_Config['y_modulus'] = [110e9,]
_Config['p_ratio'] = [0.22,]
_Config['isolation'] = 200e-9
_Config['permittivity'] = 6.3
_Config['gap'] = 50e-9
_Config['damping_mode_a'] = 0
_Config['damping_mode_b'] = 4
_Config['damping_ratio_a'] = 0.004
_Config['damping_ratio_b'] = 0.005
_Config['npatch'] = [3, 3]
# array properties
_Config['mempitch'] = [60e-6, 60e-6]
_Config['nmem'] = [1, 1]
_Config['elempitch'] = [60e-6, 60e-6]
_Config['nelem'] = [5, 5]
Config = register_type('Config', _Config)


if __name__ == '__main__':

    import sys
    from cnld import util

    # get script parser and parse arguments
    parser, run_parser = util.script_parser(main, Config)
    args = parser.parse_args()
    array = args.func(args)

    if array is not None:
        if args.file:
            dump(array, args.file)
        else:
            print(array)
    
