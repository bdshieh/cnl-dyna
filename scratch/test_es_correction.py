'''
'''
import numpy as np
import scipy as sp
from scipy.constants import epsilon_0 as e_0
from scipy.interpolate import interp1d
import numpy.linalg

from cnld import abstract, fem, mesh


def calc_es_correction(array, refn):

    fcorr = []

    for elem in array.elements:
        for mem in elem.membranes:
            if isinstance(mem, abstract.SquareCmutMembrane):
                square = True
                amesh = mesh.square(mem.length_x, mem.length_y, refn=refn)
            elif isinstance(mem, abstract.CircularCmutMembrane):
                square = False
                amesh = mesh.circle(mem.radius, refn=refn)

            K = fem.mem_k_matrix(amesh, mem.y_modulus, mem.thickness, mem.p_ratio)
            Kinv = np.linalg.inv(K)
            g = mem.gap
            g_eff = mem.gap + mem.isolation / mem.permittivity

            for pat in mem.patches:
                if square:
                    f = fem.square_patch_f_vector(amesh.vertices, amesh.triangles, amesh.on_boundary,
                        mem.length_x, mem.length_y, pat.position[0] - mem.position[0], 
                        pat.position[1] - mem.position[1], pat.length_x, pat.length_y)
                else:
                    f = fem.circular_patch_f_vector(amesh.vertices, amesh.triangles, amesh.on_boundary,
                        mem.radius, pat.position[0] - mem.position[0], pat.position[1] - mem.position[1], 
                        pat.radius_min, pat.radius_max, pat.theta_min, pat.theta_max)
                
                u = Kinv.dot(-f)
                unorm = u / np.max(np.abs(u))

                d = np.linspace(0, 1, 11)
                fc = []
                uavg = []
                for di in d:
                    fc.append((-e_0 / 2 / (unorm * g * di + g_eff)**2).dot(f) / pat.area)
                    uavg.append((unorm * g * di).dot(f) / pat.area)
                
                # fcorr.append((d, uavg, fc, fpp))
                fcorr.append(interp1d(uavg, fc, kind='cubic', bounds_error=False, fill_value=(fc[-1], fc[0])))

    return fcorr


array = abstract.load('square_membrane.json')
fcorr = calc_es_correction(array, refn=9)