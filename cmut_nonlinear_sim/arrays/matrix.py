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
    mem_properties = dict()
    mem_properties['length_x'] = length_x
    mem_properties['length_y'] = length_y
    mem_properties['electrode_x'] = electrode_x
    mem_properties['electrode_y'] = electrode_y
    mem_properties['y_modulus'] = kwargs['y_modulus']
    mem_properties['p_ratio'] = kwargs['p_ratio']
    mem_properties['isolation'] = kwargs['isolation']
    mem_properties['permittivity'] = kwargs['permittivity']
    mem_properties['gap'] = kwargs['gap']
    mem_properties['npatch_x'] = npatch_x
    mem_properties['npatch_y'] = npatch_y
    mem_properties['thickness'] = kwargs['thickness']
    mem_properties['density'] = kwargs['density']
    mem_properties['att_mech'] = kwargs['att_mech']
    mem_properties['k_matrix_comsol_file'] = kwargs['k_matrix_comsol_file']

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

    # construct elements
    elements = []
    for i, epos in enumerate(elem_pos):

        membranes = []
        
        for j, mpos in enumerate(mem_pos):

            # construct membrane
            m = SquareCmutMembrane(**mem_properties)
            m.id = i * len(mem_pos) + j
            m.position = (epos + mpos).tolist()
            membranes.append(m)

        # construct element
        elem = Element(id=i,
                       position=epos.tolist(),
                       kind='both',
                       membranes=membranes)
        element_position_from_membranes(elem)
        elements.append(elem)

    # construct array
    array = Array(id=0,
                  elements=elements,
                  position=[0, 0, 0])

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

    ##
    from cmut_nonlinear_sim.mesh import from_abstract
    mesh = from_abstract(spec)