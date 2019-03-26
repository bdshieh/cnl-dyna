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

            # F = fem.mem_f_vector(amesh, 1)
            F = np.array(fem.f_from_abstract(array, refn).todense())
            # u = Kinv.dot(-F)
            # unorm = u / np.max(np.abs(u))

            for i, pat in enumerate(mem.patches):
                if square:
                    avg = fem.square_patch_avg_vector(amesh.vertices, amesh.triangles, amesh.on_boundary,
                        mem.length_x, mem.length_y, pat.position[0] - mem.position[0], 
                        pat.position[1] - mem.position[1], pat.length_x, pat.length_y)
                else:
                    avg = fem.circular_patch_avg_vector(amesh.vertices, amesh.triangles, amesh.on_boundary,
                        mem.radius, pat.position[0] - mem.position[0], pat.position[1] - mem.position[1], 
                        pat.radius_min, pat.radius_max, pat.theta_min, pat.theta_max)

                u = Kinv.dot(-F[:,i])
                unorm = u / np.max(np.abs(u))

                d = np.linspace(0, 1, 11)
                fc = []
                uavg = []
                fpp = []
                for di in d:
                    ubar = (unorm * g * di).dot(avg) / pat.area
                    uavg.append(ubar)

                    fc.append((-e_0 / 2 / (unorm * g * di + g_eff)**2).dot(avg) / pat.area)
                    fpp.append(-e_0 / 2 / (ubar + g_eff)**2)

                fcorr.append((d, unorm, uavg, fc, fpp))
                # fcorr.append(interp1d(uavg, fc, kind='cubic', bounds_error=False, fill_value=(fc[-1], fc[0])))

    return fcorr


array = abstract.load('circular_membrane.json')
fcorr = calc_es_correction(array, refn=9)



from matplotlib import pyplot as plt

for i in [0, 4, 8]:

    d, unorm, uavg, fc, fpp = fcorr[i]

    fig, ax = plt.subplots()
    ax.plot(uavg, fc)
    ax.plot(uavg, fpp, '--')
    fig.show()

# amesh = mesh.Mesh.from_abstract(array, 9)
# f1 = fem.mem_f_vector(amesh, 1)
# f2 = fem.mem_f_vector2(amesh, 1)