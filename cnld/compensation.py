'''
'''
import numpy as np
import scipy as sp
from scipy.constants import epsilon_0 as e_0
from scipy.interpolate import interp1d
import numpy.linalg

from cnld import abstract, mesh, fem, util


@util.memoize
def patch_fcomp(patch):

    f = fem.square_patch_f_vector(amesh.vertices, amesh.triangles, amesh.on_boundary,
            mem.length_x, mem.length_y, pat.position[0] - mem.position[0], 
            pat.position[1] - mem.position[1], pat.length_x, pat.length_y)
    avg = fem.square_patch_avg_vector()

    u = Kinv.dot(-f)
    unorm = u / np.max(np.abs(u))

    d = np.linspace(0, 1, 11)
    fc = []
    uavg = []
    for di in d:
        ubar = (unorm * g * di).dot(avg) / pat.area
        uavg.append(ubar)
        fc.append((-e_0 / 2 / (unorm * g * di + g_eff)**2).dot(avg) / pat.area)

    fcomp = interp1d(uavg, fc, kind='cubic', bounds_error=False, fill_value=(fc[-1], fc[0]))


@util.memoize
def mem_static_x(amesh, E, h, eta, p):

    K = fem.mem_k_matrix(amesh, E, h, eta)
    Kinv = np.linalg.inv(K)
    f = fem.mem_f_vector(amesh, p)

    x = Kinv.dot(f)
    return x


def fcomp_from_abstract(array, refn):
    '''
    Construct load vector based on patches of abstract array.
    '''
    blocks = []
    for elem in array.elements:
        for mem in elem.membranes:
            if isinstance(mem, abstract.SquareCmutMembrane):
                square = True
                amesh = mesh.square(mem.length_x, mem.length_y, refn=refn)
            elif isinstance(mem, abstract.CircularCmutMembrane):
                square = False
                amesh = mesh.circle(mem.radius, refn=refn)
            else:
                raise ValueError

            f = np.zeros((len(amesh.vertices), len(mem.patches)))
            for i, pat in enumerate(mem.patches):
                if square:
                    f[:,i] = square_patch_f_vector(amesh.vertices, amesh.triangles, amesh.on_boundary,
                        mem.length_x, mem.length_y, pat.position[0] - mem.position[0], 
                        pat.position[1] - mem.position[1], pat.length_x, pat.length_y)
                else:
                    f[:,i] = circular_patch_f_vector(amesh.vertices, amesh.triangles, amesh.on_boundary,
                        mem.radius, pat.position[0] - mem.position[0], pat.position[1] - mem.position[1], 
                        pat.radius_min, pat.radius_max, pat.theta_min, pat.theta_max)

                f[amesh.on_boundary,i] = 0
            blocks.append(f)
    
    return sps.block_diag(blocks, format='csc')


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
        unorm = u / np.max(np.abs(u))
        ubar = unorm.dot(avg) / pat.area

        fc = []
        uavg = []
        for d in np.linspace(0, 1, 6):
            uavg.append(ubar * g * di)
            x = unorm * g * di
            fc.append((-e_0 / 2 / (x + g_eff)**2).dot(avg) / pat.area)

        for di in np.linspace(1.2, 2, 5):
            x = unorm * g * di
            x[x > g] = g
            uavg.append(x.dot(avg) / pat.area)
            fc.append((-e_0 / 2 / (x + g_eff)**2).dot(avg) / pat.area)
            
        # fcomp = interp1d(uavg, fc, kind='cubic', bounds_error=False, fill_value=(fc[-1], fc[0]))
        fcomp = uavg, fc
        fcomps.append(fcomp)

    return fcorr