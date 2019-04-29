'''
'''
import numpy as np
import scipy as sp
from scipy.constants import epsilon_0 as e_0
from scipy.interpolate import interp1d, CubicSpline
import numpy.linalg

from cnld import abstract, mesh, fem, util


@util.memoize
def mem_patch_fcomp_funcs(mem, refn):
    '''
    '''
    f = fem.mem_patch_f_matrix(mem, refn)
    avg = fem.mem_patch_avg_matrix(mem, refn)

    K = fem.mem_k_matrix(mem, refn)
    Kinv = fem.inv_block(K)

    g = mem.gap
    g_eff = mem.gap + mem.isolation / mem.permittivity

    u = Kinv.dot(-np.sum(f, axis=1)).squeeze()
    unorm = u / np.max(np.abs(u))
    
    fcomps = []

    for i, pat in enumerate(mem.patches):

        avg_pat = avg[:,i]

        u_pat = unorm.dot(avg_pat)

        fc = np.zeros(11)
        uavg = np.zeros(11)

        for i, d in enumerate(np.linspace(-1, 1, 11)):
            uavg[i] = u_pat * g * d
            x = unorm * g * d
            fc[i] = (-e_0 / 2 / (x + g_eff)**2).dot(avg_pat)

        uavg = np.append(uavg, -g)
        fc = np.append(fc, -e_0 / 2 / (-g + g_eff)**2)

        fcomp = CubicSpline(uavg[::-1], fc[::-1], bc_type=((1, 0),'not-a-knot'))
        fcomps.append(fcomp)
    
    return fcomps


def array_patch_fcomp_funcs(array, refn):
    '''
    '''
    fcomps = []
    for elem in array.elements:
        for mem in elem.membranes:
            fcomps += mem_patch_fcomp_funcs(mem, refn)
            
    return fcomps


def fcomp_from_abstract(array, refn):

    F = fem.array_f_spmatrix(array, refn)
    AVG = fem.array_avg_spmatrix(array, refn)

    patches = abstract.get_patches_from_array(array)

    gaps = []
    gaps_eff = []
    Kinvs = []
    for elem in array.elements:
        for mem in elem.membranes:
            # if isinstance(mem, abstract.SquareCmutMembrane):
            #     square = True
            #     amesh = mesh.square(mem.length_x, mem.length_y, refn=refn)
            # elif isinstance(mem, abstract.CircularCmutMembrane):
            #     square = False
            #     amesh = mesh.circle(mem.radius, refn=refn)

            # K = fem.mem_k_matrix(amesh, mem.y_modulus, mem.thickness, mem.p_ratio)
            K = fem.mem_k_matrix(mem, refn)
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
        ubar = unorm.dot(avg) 

        fc = np.zeros(11)
        uavg = np.zeros(11)

        for i, d in enumerate(np.linspace(0, 1, 11)):
            uavg[i] = ubar * g * d
            x = unorm * g * d
            fc[i] = (-e_0 / 2 / (x + g_eff)**2).dot(avg)

        uavg = np.append(uavg, -g)
        fc = np.append(fc, -e_0 / 2 / (-g + g_eff)**2)

        fcomp = CubicSpline(uavg[::-1], fc[::-1], bc_type=((1, 0),'not-a-knot'))
        fcomps.append(fcomp)

    return fcomps