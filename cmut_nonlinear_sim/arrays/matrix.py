''' Abstract representation of a matrix array.
'''
import numpy as np

from cmut_nonlinear_sim.abstract import *

# default parameters
defaults = {}
# membrane properties
defaults['length'] = [40e-6, 40e-6]
defaults['electrode'] = [40e-6, 40e-6]
defaults['thickness'] = [2e-6,]
defaults['density'] = [2040,]
defaults['y_modulus'] = [110e9,]
defaults['p_ratio'] = [0.22,]
defaults['isolation'] = 200e-9
defaults['permittivity'] = 6.3
defaults['gap'] = 100e-9
defaults['att_mech'] = 0
defaults['npatch'] = [3, 3]
defaults['k_matrix_comsol_file'] = None
# array properties
defaults['mempitch'] = [60e-6, 60e-6]
defaults['nmem'] = [1, 1]
defaults['elempitch'] = [60e-6, 60e-6]
defaults['nelem'] = [5, 5]


def init(**kwargs):

    # set defaults if not in kwargs:
    for k, v in defaults.items():
        kwargs.setdefault(k, v)

    nmem_x, nmem_y = kwargs['nmem']
    mempitch_x, mempitch_y = kwargs['mempitch']
    length_x, length_y = kwargs['length']
    electrode_x, electrode_y = kwargs['electrode']
    npatch_x, npatch_y = kwargs['npatch']
    nelem_x, nelem_y = kwargs['nelem']
    elempitch_x, elempitch_y = kwargs['elempitch']

    # membrane properties
    memprops = {}
    memprops['length_x'] = length_x
    memprops['length_y'] = length_y
    memprops['electrode_x'] = electrode_x
    memprops['electrode_y'] = electrode_y
    memprops['y_modulus'] = kwargs['y_modulus']
    memprops['p_ratio'] = kwargs['p_ratio']
    memprops['isolation'] = kwargs['isolation']
    memprops['permittivity'] = kwargs['permittivity']
    memprops['gap'] = kwargs['gap']
    memprops['npatch_x'] = npatch_x
    memprops['npatch_y'] = npatch_y
    memprops['thickness'] = kwargs['thickness']
    memprops['density'] = kwargs['density']
    memprops['att_mech'] = kwargs['att_mech']
    memprops['k_matrix_comsol_file'] = kwargs['k_matrix_comsol_file']

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


## COMMAND LINE INTERFACE ##

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-nmem', '--nmem', nargs=2, type=int)
    parser.add_argument('-mempitch', '--mempitch', nargs=2, type=float)
    parser.add_argument('-nelem', '--nelem', nargs=2, type=int)
    parser.add_argument('-elempitch', '--elempitch', nargs=2, type=float)
    parser.add_argument('-d', '--dump', nargs='?', default=None)
    parser.set_defaults(**defaults)

    args = vars(parser.parse_args())
    filename = args.pop('dump')

    spec = init(**args)
    print(spec)

    if filename is not None:
        dump(spec, filename)

    # from cmut_nonlinear_sim.mesh import Mesh
    # mesh = Mesh.from_abstract(spec, refn=3)