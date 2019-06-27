'''
'''
import numpy as np

import cnld.scripts.matrix_array


def matrix_array(**kwargs):
    return cnld.scripts.matrix_array.main(cnld.scripts.matrix_array.Config(**kwargs), None)


def hex_array(**kwargs):
    return cnld.scripts.hex_array.main(cnld.scripts.hex_array.Config(**kwargs), None)


