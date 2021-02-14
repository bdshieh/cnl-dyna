''''''
import numpy as np
from matplotlib import pyplot as plt
from cnld.api import define

geom = define.circle_cmut_1mhz_geometry()
lay = define.matrix_layout(nx=2, ny=2, pitch_x=100e-6, pitch_y=100e-6)
lay.controldomains = define.generate_control_domains(lay, geom)
