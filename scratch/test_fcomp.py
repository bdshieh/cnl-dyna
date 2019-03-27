

import numpy as np
import scipy as sp
from scipy.constants import epsilon_0 as e_0
from scipy.interpolate import interp1d
import numpy.linalg

from cnld import abstract, mesh, fem, util


def fcomp_from_abstract(array, refn):

    F = fem.f_from_abstract(array, refn)
    AVG = fem.avg_from_abstract(array, refn)

    patches = abstract.get_patches_from_array(array)

    gaps = []
    gaps_eff = []
    Kinvs = []
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

            # f = fem.mem_f_vector(amesh, 1)
            # u = Kinv.dot(-f)
            # unorm = u / np.max(np.abs(u))

            for pat in mem.patches:
                Kinvs.append(Kinv)
                gaps.append(mem.gap)
                gaps_eff.append(mem.gap + mem.isolation / mem.permittivity)

    fcomps = []
    for i, pat in enumerate(patches):
        
        f = np.array(F[:,i].todense())
        avg = np.array(AVG[:,i].todense())
        g = gaps[i]
        g_eff = gaps_eff[i]
        Kinv = Kinvs[i]

        u = Kinv.dot(-f)
        unorm = (u / np.max(np.abs(u))).squeeze()
        ubar = unorm.dot(avg) / pat.area

        fc = []
        uavg = []
        fpp = []
        for d in np.linspace(0, 1, 11):
            uavg.append(ubar * g * d)
            x = unorm * g * d
            fc.append((-e_0 / 2 / (x + g_eff)**2).dot(avg) / pat.area)
            fpp.append(-e_0 / 2 / (ubar * g * d + g_eff)**2)

        for d in np.linspace(1.2, 3, 10):
            x = unorm * g * d
            x[x < -g] = -g
            xbar = x.dot(avg) / pat.area
            uavg.append(x.dot(avg) / pat.area)
            fc.append((-e_0 / 2 / (x + g_eff)**2).dot(avg) / pat.area)
            fpp.append(-e_0 / 2 / (xbar + g_eff)**2)
            
        # fcomp = interp1d(uavg, fc, kind='cubic', bounds_error=False, fill_value=(fc[-1], fc[0]))
        fcomp = uavg, fc, fpp
        fcomps.append(fcomp)

    return fcomps

# array = abstract.load('square_membrane.json')
array = abstract.load('circular_membrane.json')
fcorr = fcomp_from_abstract(array, refn=9)



from matplotlib import pyplot as plt

# for i in [4, 5, 8]:
for i in [0, 4, 8]:

    uavg, fc, fpp = fcorr[i]

    fig, ax = plt.subplots()
    ax.plot(uavg, fc,'.-')
    ax.plot(uavg, fpp, '--')
    fig.show()