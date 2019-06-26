''' 
Abstract representation of a matrix array.
'''
import numpy as np

from cnld.abstract import *
from cnld import util


def main(cfg, args):

    radius = cfg.radius
    electrode_r = cfg.electrode_r
    npatch_r, npatch_theta = cfg.npatch
    nx, ny = cfg.nelem
    pitch = cfg.elempitch

    # membrane properties
    memprops = {}
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
    patch_r = np.linspace(0, radius, npatch_r + 1)
    patch_theta = np.linspace(-np.pi, np.pi, npatch_theta + 1)
    patch_rmin = [patch_r[i] for i in range(npatch_r) for j in range(npatch_theta)]
    patch_rmax = [patch_r[i + 1] for i in range(npatch_r) for j in range(npatch_theta)]
    patch_thetamin = [patch_theta[j] for i in range(npatch_r) for j in range(npatch_theta)]
    patch_thetamax = [patch_theta[j + 1] for i in range(npatch_r) for j in range(npatch_theta)]
    patch_pos = np.array([0, 0, 0])

    # calculate element positions
    pitch_x = np.sqrt(3) / 2 * pitch
    pitch_y = pitch
    offset_y = pitch / 2

    xx, yy, zz = np.meshgrid(np.linspace(0, (nx - 1) * pitch_x, nx),
                             np.linspace(0, (ny - 1) * pitch_y, ny),
                             0)
    yy[:, ::2, :] += offset_y / 2
    yy[:, 1::2, :] -= offset_y / 2

    elem_pos = np.c_[xx.ravel(), yy.ravel(), zz.ravel()] - [(nx - 1) * pitch_x / 2,
                                                           (ny - 1) * pitch_y / 2,
                                                           0]

    # construct element list
    elements = []
    elem_counter = 0
    mem_counter = 0
    patch_counter = 0

    for epos in elem_pos:
        # construct membrane list
        membranes = []
        # construct patch list
        patches = []
        for i in range(npatch_r * npatch_theta):
            p = Patch()
            p.id = patch_counter
            p.position = (epos + patch_pos).tolist()
            p.radius_min = patch_rmin[i]
            p.radius_max = patch_rmax[i]
            p.theta_min = patch_thetamin[i]
            p.theta_max = patch_thetamax[i]
            p.area = (p.radius_max**2 - p.radius_min**2) * (p.theta_max - p.theta_min) / 2

            patches.append(p)
            patch_counter += 1

        # construct membrane
        m = CircularCmutMembrane(**memprops)
        m.id = mem_counter
        m.position = epos.tolist()
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
_Config['damping_ratio_a'] = 0.0
_Config['damping_ratio_b'] = 0.0
_Config['npatch'] = [2, 4]
# array properties
_Config['elempitch'] = 60e-6
_Config['nelem'] = [2, 2]
Config = register_type('Config', _Config)


if __name__ == '__main__':

    import sys
    from cnld import util

    # get script parser and parse arguments
    parser = util.script_parser2(main, Config)
    args = parser.parse_args()
    array = args.func(args)

    if array is not None:
        if args.file:
            dump(array, args.file)
        else:
            print(array)
    
