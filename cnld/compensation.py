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

        avg_pat = avg[:, i]

        u_pat = unorm.dot(avg_pat)

        fc = np.zeros(11)
        uavg = np.zeros(11)

        for i, d in enumerate(np.linspace(-1, 1, 11)):
            uavg[i] = u_pat * g * d
            x = unorm * g * d
            fc[i] = (-e_0 / 2 / (x + g_eff)**2).dot(avg_pat)

        uavg = np.append(uavg, -g)
        fc = np.append(fc, -e_0 / 2 / (-g + g_eff)**2)

        fcomp = CubicSpline(uavg[::-1], fc[::-1], bc_type=((1, 0),
                            'not-a-knot'))
        fcomps.append(fcomp)
    
    return fcomps


@util.memoize
def mem_patch_fcomp_funcs2(mem, refn):
    '''
    '''
    f = fem.mem_patch_f_matrix(mem, refn)
    avg = fem.mem_patch_avg_matrix(mem, refn)

    K = fem.mem_k_matrix(mem, refn)
    Kinv = fem.inv_block(K)

    g = mem.gap
    g_eff = mem.gap + mem.isolation / mem.permittivity

    u = Kinv.dot(-np.sum(f, axis=1)).squeeze()
    u = u / np.max(np.abs(u))
    
    fcomps = []
    fcomps_data = []

    for i, pat in enumerate(mem.patches):

        avg_pat = avg[:, i]
        unorm = u / np.max(np.abs(u[avg_pat > 0]))
        u_pat = unorm.dot(avg_pat)

        fc = []
        uavg = []

        for i, d in enumerate(np.linspace(-2, 2, 21)):
            x = unorm * g * d
            x[x < -g] = -g

            # uavg.append(u_pat * g * d)
            uavg.append(x.dot(avg_pat))
            fc.append((-e_0 / 2 / (x + g_eff)**2).dot(avg_pat))

            if np.all(x[avg_pat > 0] <= -g):
                break

        # uavg = np.append(uavg, -g)
        # fc = np.append(fc, -e_0 / 2 / (-g + g_eff)**2)
        fc = np.array(fc)
        uavg = np.array(uavg)

        fcomp = CubicSpline(uavg[::-1], fc[::-1], bc_type=((1, 0),
                            'not-a-knot'))
        fcomps.append(fcomp)
        fcomps_data.append((uavg, fc))
    
    # return fcomps, fcomps_data
    return fcomps


@util.memoize
def mem_patch_fcomp_funcs3(mem, refn, cont_stiff=None):
    '''
    '''
    f = fem.mem_patch_f_matrix(mem, refn)
    avg = fem.mem_patch_avg_matrix(mem, refn)

    K = fem.mem_k_matrix(mem, refn)
    Kinv = fem.inv_block(K)

    g = mem.gap
    g_eff = mem.gap + mem.isolation / mem.permittivity

    u = Kinv.dot(-np.sum(f, axis=1)).squeeze()
    u = u / np.max(np.abs(u))
    
    fcomps = []
    fcomps_data = []

    for i, pat in enumerate(mem.patches):
        
        if not cont_stiff:
            Estar = mem.y_modulus[0] / (2 * (1 - mem.p_ratio[0]**2))
            cont_stiff = 2 * Estar / np.sqrt(np.pi * pat.area) / 2

        avg_pat = avg[:, i]
        unorm = u / np.max(np.abs(u[avg_pat > 0]))
        # u_pat = unorm.dot(avg_pat)

        f_es = []
        f_cont = []
        uavg = []
        umax = []

        for i, d in enumerate(np.linspace(-2, 2, 81)):
            x = unorm * g * d
            x_es = x.copy()
            x_es[x_es < -g] = -g

            _avg = x_es.dot(avg_pat)
            _max = np.abs(x_es[avg_pat > 0]).max() * np.sign(-d)
            _f_es = -e_0 / 2 / (x_es + g_eff)**2
            _f_cont = np.zeros(len(x))
            _f_cont[x < -g] = -cont_stiff * (x[x < -g] + g)

            if i > 0:
                if _avg == uavg[i - 1]:
                    break

            uavg.append(_avg)
            umax.append(_max)
            f_es.append(_f_es.dot(avg_pat))
            f_cont.append(_f_cont.dot(avg_pat))

        f_es = np.array(f_es)[::-1]
        f_cont = np.array(f_cont)[::-1]
        uavg = np.array(uavg)[::-1]
        umax = np.array(umax)[::-1]

        # fcomp1 = CubicSpline(uavg[::-1], f_es[::-1], bc_type=((1, 0),
        #                      'not-a-knot'))
        # fcomp2 = CubicSpline(uavg[::-1], f_cont[::-1], bc_type=((1, 0),
        #                      'not-a-knot'))
        fcomp1 = CubicSpline(uavg, f_es, extrapolate=False)
        fcomp2 = CubicSpline(uavg, f_cont, extrapolate=False)
        # fcomp3 = CubicSpline(uavg, umax)

        fcomps.append(make_fcomp(fcomp1, fcomp2))
        fcomps_data.append({'u': uavg, 'u_max': umax, 'f_es': f_es,
                            'f_cont': f_cont})

    return fcomps, fcomps_data


@util.memoize
def mem_patch_fcomp_funcs4(mem, refn, pmax, damping):
    '''
    '''
    f = fem.mem_patch_f_matrix(mem, refn)
    avg = fem.mem_patch_avg_matrix(mem, refn)

    K = fem.mem_k_matrix(mem, refn)
    Kinv = fem.inv_block(K)

    g = mem.gap
    g_eff = mem.gap + mem.isolation / mem.permittivity

    u = Kinv.dot(-np.sum(f, axis=1)).squeeze()
    u = u / np.max(np.abs(u))
    
    # define contact pressure function
    
    def make_fcontact(damping):

        cs = CubicSpline([-g, -g + 5e-9], [pmax, 0], bc_type=('natural', 'clamped'))
        
        def _fcontact(x, xdot):
            if x > -45e-9:
                return 0
            return cs(x) - damping * xdot
        return np.vectorize(_fcontact)

    fcontact = make_fcontact(damping)

    # define electrostatic pressure function
    def make_fes(umin, fmin, cs):
        def _fes(x, v):
            if x < umin:
                return fmin * v**2
            return cs(x) * v**2
        return np.vectorize(_fes)

    fcomps = []

    for i, pat in enumerate(mem.patches):
        
        avg_pat = avg[:, i]
        unorm = u / np.max(np.abs(u[avg_pat > 0]))

        f_es = []
        uavg = []
        umax = []

        for i, d in enumerate(np.linspace(-2, 2, 81)):
            x = unorm * g * d
            x_es = x.copy()
            x_es[x_es < -g] = -g

            _avg = x_es.dot(avg_pat)
            _max = np.abs(x_es[avg_pat > 0]).max() * np.sign(-d)
            _f_es = -e_0 / 2 / (x_es + g_eff)**2

            if i > 0:
                if _avg == uavg[i - 1]:
                    break

            uavg.append(_avg)
            umax.append(_max)
            f_es.append(_f_es.dot(avg_pat))

        f_es = np.array(f_es)[::-1]
        uavg = np.array(uavg)[::-1]
        umax = np.array(umax)[::-1]
        cs = CubicSpline(uavg, f_es, extrapolate=False)

        fes = make_fes(np.min(uavg), np.min(f_es), cs)
        
        fcomps.append({'fes': fes, 'fcontact': fcontact})

    return fcomps


@util.memoize
def mem_patch_fcomp_funcs5(mem, refn, lmbd, k, n, x0):
    '''
    '''
    f = fem.mem_patch_f_matrix(mem, refn)
    avg = fem.mem_patch_avg_matrix(mem, refn)

    K = fem.mem_k_matrix(mem, refn)
    Kinv = fem.inv_block(K)

    g = mem.gap
    g_eff = mem.gap + mem.isolation / mem.permittivity

    u = Kinv.dot(-np.sum(f, axis=1)).squeeze()
    u = u / np.max(np.abs(u))
    
    # define electrostatic pressure function
    def make_fes(umin, fmin, cs):
        def _fes(x, v):
            if x < umin:
                return fmin * v**2
            return cs(x) * v**2
        return np.vectorize(_fes)

    fcomps = []

    for i, pat in enumerate(mem.patches):
        
        avg_pat = avg[:, i]
        unorm = u / np.max(np.abs(u[avg_pat > 0]))

        f_es = []
        uavg = []
        umax = []

        for i, d in enumerate(np.linspace(-2, 2, 81)):
            x = unorm * g * d
            x_es = x.copy()
            x_es[x_es < -g] = -g

            _avg = x_es.dot(avg_pat)
            _max = np.abs(x_es[avg_pat > 0]).max() * np.sign(-d)
            _f_es = -e_0 / 2 / (x_es + g_eff)**2

            if i > 0:
                if _avg == uavg[i - 1]:
                    break

            uavg.append(_avg)
            umax.append(_max)
            f_es.append(_f_es.dot(avg_pat))

        f_es = np.array(f_es)[::-1]
        uavg = np.array(uavg)[::-1]
        umax = np.array(umax)[::-1]
        cs = CubicSpline(uavg, f_es, extrapolate=False)

        fes = make_fes(np.min(uavg), np.min(f_es), cs)
        
        fcomps.append({'fes': fes, 'fcontact': fcontact})

    return fcomps


def make_fcomp(fcomp1, fcomp2):
    def fcomp(x, v):
        return fcomp1(x) * v**2 + fcomp2(x)
    fcomp.fcomp1 = fcomp1
    fcomp.fcomp2 = fcomp2
    return fcomp


def array_patch_fcomp_funcs(array, refn, **kwargs):
    '''
    '''
    fcomps = []
    for elem in array.elements:
        for mem in elem.membranes:
            fcomps += mem_patch_fcomp_funcs5(mem, refn, **kwargs)
            
    return fcomps


# def fcomp_from_abstract(array, refn):

#     F = fem.array_f_spmatrix(array, refn)
#     AVG = fem.array_avg_spmatrix(array, refn)

#     patches = abstract.get_patches_from_array(array)

#     gaps = []
#     gaps_eff = []
#     Kinvs = []
#     for elem in array.elements:
#         for mem in elem.membranes:
#             # if isinstance(mem, abstract.SquareCmutMembrane):
#             #     square = True
#             #     amesh = mesh.square(mem.length_x, mem.length_y, refn=refn)
#             # elif isinstance(mem, abstract.CircularCmutMembrane):
#             #     square = False
#             #     amesh = mesh.circle(mem.radius, refn=refn)

#             # K = fem.mem_k_matrix(amesh, mem.y_modulus, mem.thickness, mem.p_ratio)
#             K = fem.mem_k_matrix(mem, refn)
#             Kinv = np.linalg.inv(K)

#             # f = fem.mem_f_vector(amesh, 1)
#             # u = Kinv.dot(-f)
#             # unorm = u / np.max(np.abs(u))

#             for pat in mem.patches:
#                 Kinvs.append(Kinv)
#                 gaps.append(mem.gap)
#                 gaps_eff.append(mem.gap + mem.isolation / mem.permittivity)

#     fcomps = []
#     for i, pat in enumerate(patches):
        
#         f = np.array(F[:,i].todense())
#         avg = np.array(AVG[:,i].todense())
#         g = gaps[i]
#         g_eff = gaps_eff[i]
#         Kinv = Kinvs[i]

#         u = Kinv.dot(-f)
#         unorm = (u / np.max(np.abs(u))).squeeze()
#         ubar = unorm.dot(avg) 

#         fc = np.zeros(11)
#         uavg = np.zeros(11)

#         for i, d in enumerate(np.linspace(0, 1, 11)):
#             uavg[i] = ubar * g * d
#             x = unorm * g * d
#             fc[i] = (-e_0 / 2 / (x + g_eff)**2).dot(avg)

#         uavg = np.append(uavg, -g)
#         fc = np.append(fc, -e_0 / 2 / (-g + g_eff)**2)

#         fcomp = CubicSpline(uavg[::-1], fc[::-1], bc_type=((1, 0),'not-a-knot'))
#         fcomps.append(fcomp)

#     return fcomps