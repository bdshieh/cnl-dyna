'''
Convenience module wrapping array scripts with functions.
'''
import cnld.scripts.hex_array
import cnld.scripts.matrix_array
import numpy as np


def matrix_array(**kwargs):
    '''
    Array of membranes arranged on a matrix (cartesian) grid.
    '''
    return cnld.scripts.matrix_array.main(cnld.scripts.matrix_array.Config(**kwargs),
                                          None)


# add script configuration to function for easier access
matrix_array.Config = cnld.scripts.matrix_array.Config


def hex_array(**kwargs):
    '''
    Array of membranes arranged on a hexagonal grid.
    '''
    return cnld.scripts.hex_array.main(cnld.scripts.hex_array.Config(**kwargs), None)


# add script configuration to function for easier access
hex_array.Config = cnld.scripts.hex_array.Config
