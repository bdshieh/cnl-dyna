'''
'''
import numpy as np
import scipy as sp
from scipy.constants import epsilon_0 as e_0
from scipy.interpolate import interp1d, CubicSpline
import numpy.linalg

from cnld import abstract, mesh, fem, util

# @util.memoize
# def mem_patch_fcomp_funcs(mem, refn):
#     '''
#     '''
#     f = fem.mem_patch_f_matrix(mem, refn)
#     avg = fem.mem_patch_avg_matrix(mem, refn)

#     K = fem.mem_k_matrix(mem, refn)
#     Kinv = fem.inv_block(K)

#     g = mem.gap
#     g_eff = mem.gap + mem.isolation / mem.permittivity

#     u = Kinv.dot(-np.sum(f, axis=1)).squeeze()
#     unorm = u / np.max(np.abs(u))

#     fcomps = []

#     for i, pat in enumerate(mem.patches):

#         avg_pat = avg[:, i]

#         u_pat = unorm.dot(avg_pat)

#         fc = np.zeros(11)
#         uavg = np.zeros(11)

#         for i, d in enumerate(np.linspace(-1, 1, 11)):
#             uavg[i] = u_pat * g * d
#             x = unorm * g * d
#             fc[i] = (-e_0 / 2 / (x + g_eff)**2).dot(avg_pat)

#         uavg = np.append(uavg, -g)
#         fc = np.append(fc, -e_0 / 2 / (-g + g_eff)**2)

#         fcomp = CubicSpline(uavg[::-1], fc[::-1], bc_type=((1, 0),
#                                                            'not-a-knot'))
#         fcomps.append(fcomp)

#     return fcomps

# @util.memoize
# def mem_patch_fcomp_funcs2(mem, refn):
#     '''
#     '''
#     f = fem.mem_patch_f_matrix(mem, refn)
#     avg = fem.mem_patch_avg_matrix(mem, refn)

#     K = fem.mem_k_matrix(mem, refn)
#     Kinv = fem.inv_block(K)

#     g = mem.gap
#     g_eff = mem.gap + mem.isolation / mem.permittivity

#     u = Kinv.dot(-np.sum(f, axis=1)).squeeze()
#     u = u / np.max(np.abs(u))

#     fcomps = []
#     fcomps_data = []

#     for i, pat in enumerate(mem.patches):

#         avg_pat = avg[:, i]
#         unorm = u / np.max(np.abs(u[avg_pat > 0]))
#         u_pat = unorm.dot(avg_pat)

#         fc = []
#         uavg = []

#         for i, d in enumerate(np.linspace(-2, 2, 21)):
#             x = unorm * g * d
#             x[x < -g] = -g

#             # uavg.append(u_pat * g * d)
#             uavg.append(x.dot(avg_pat))
#             fc.append((-e_0 / 2 / (x + g_eff)**2).dot(avg_pat))

#             if np.all(x[avg_pat > 0] <= -g):
#                 break

#         # uavg = np.append(uavg, -g)
#         # fc = np.append(fc, -e_0 / 2 / (-g + g_eff)**2)
#         fc = np.array(fc)
#         uavg = np.array(uavg)

#         fcomp = CubicSpline(uavg[::-1], fc[::-1], bc_type=((1, 0),
#                                                            'not-a-knot'))
#         fcomps.append(fcomp)
#         fcomps_data.append((uavg, fc))

#     # return fcomps, fcomps_data
#     return fcomps

# @util.memoize
# def mem_patch_fcomp_funcs3(mem, refn, cont_stiff=None):
#     '''
#     '''
#     f = fem.mem_patch_f_matrix(mem, refn)
#     avg = fem.mem_patch_avg_matrix(mem, refn)

#     K = fem.mem_k_matrix(mem, refn)
#     Kinv = fem.inv_block(K)

#     g = mem.gap
#     g_eff = mem.gap + mem.isolation / mem.permittivity

#     u = Kinv.dot(-np.sum(f, axis=1)).squeeze()
#     u = u / np.max(np.abs(u))

#     fcomps = []
#     fcomps_data = []

#     for i, pat in enumerate(mem.patches):

#         if not cont_stiff:
#             Estar = mem.y_modulus[0] / (2 * (1 - mem.p_ratio[0]**2))
#             cont_stiff = 2 * Estar / np.sqrt(np.pi * pat.area) / 2

#         avg_pat = avg[:, i]
#         unorm = u / np.max(np.abs(u[avg_pat > 0]))
#         # u_pat = unorm.dot(avg_pat)

#         f_es = []
#         f_cont = []
#         uavg = []
#         umax = []

#         for i, d in enumerate(np.linspace(-2, 2, 81)):
#             x = unorm * g * d
#             x_es = x.copy()
#             x_es[x_es < -g] = -g

#             _avg = x_es.dot(avg_pat)
#             _max = np.abs(x_es[avg_pat > 0]).max() * np.sign(-d)
#             _f_es = -e_0 / 2 / (x_es + g_eff)**2
#             _f_cont = np.zeros(len(x))
#             _f_cont[x < -g] = -cont_stiff * (x[x < -g] + g)

#             if i > 0:
#                 if _avg == uavg[i - 1]:
#                     break

#             uavg.append(_avg)
#             umax.append(_max)
#             f_es.append(_f_es.dot(avg_pat))
#             f_cont.append(_f_cont.dot(avg_pat))

#         f_es = np.array(f_es)[::-1]
#         f_cont = np.array(f_cont)[::-1]
#         uavg = np.array(uavg)[::-1]
#         umax = np.array(umax)[::-1]

#         # fcomp1 = CubicSpline(uavg[::-1], f_es[::-1], bc_type=((1, 0),
#         #                      'not-a-knot'))
#         # fcomp2 = CubicSpline(uavg[::-1], f_cont[::-1], bc_type=((1, 0),
#         #                      'not-a-knot'))
#         fcomp1 = CubicSpline(uavg, f_es, extrapolate=False)
#         fcomp2 = CubicSpline(uavg, f_cont, extrapolate=False)
#         # fcomp3 = CubicSpline(uavg, umax)

#         fcomps.append(make_fcomp(fcomp1, fcomp2))
#         fcomps_data.append({'u': uavg, 'u_max': umax, 'f_es': f_es,
#                             'f_cont': f_cont})

#     return fcomps, fcomps_data

# @util.memoize
# def mem_patch_fcomp_funcs4(mem, refn, pmax, damping):
#     '''
#     '''
#     f = fem.mem_patch_f_matrix(mem, refn)
#     avg = fem.mem_patch_avg_matrix(mem, refn)

#     K = fem.mem_k_matrix(mem, refn)
#     Kinv = fem.inv_block(K)

#     g = mem.gap
#     g_eff = mem.gap + mem.isolation / mem.permittivity

#     u = Kinv.dot(-np.sum(f, axis=1)).squeeze()
#     u = u / np.max(np.abs(u))

#     # define contact pressure function

#     def make_fcontact(damping):

#         cs = CubicSpline([-g, -g + 5e-9], [pmax, 0],
#                          bc_type=('natural', 'clamped'))

#         def _fcontact(x, xdot):
#             if x > -45e-9:
#                 return 0
#             return cs(x) - damping * xdot
#         return np.vectorize(_fcontact)

#     fcontact = make_fcontact(damping)

#     # define electrostatic pressure function
#     def make_fes(umin, fmin, cs):
#         def _fes(x, v):
#             if x < umin:
#                 return fmin * v**2
#             return cs(x) * v**2
#         return np.vectorize(_fes)

#     fcomps = []

#     for i, pat in enumerate(mem.patches):

#         avg_pat = avg[:, i]
#         unorm = u / np.max(np.abs(u[avg_pat > 0]))

#         f_es = []
#         uavg = []
#         umax = []

#         for i, d in enumerate(np.linspace(-2, 2, 81)):
#             x = unorm * g * d
#             x_es = x.copy()
#             x_es[x_es < -g] = -g

#             _avg = x_es.dot(avg_pat)
#             _max = np.abs(x_es[avg_pat > 0]).max() * np.sign(-d)
#             _f_es = -e_0 / 2 / (x_es + g_eff)**2

#             if i > 0:
#                 if _avg == uavg[i - 1]:
#                     break

#             uavg.append(_avg)
#             umax.append(_max)
#             f_es.append(_f_es.dot(avg_pat))

#         f_es = np.array(f_es)[::-1]
#         uavg = np.array(uavg)[::-1]
#         umax = np.array(umax)[::-1]
#         cs = CubicSpline(uavg, f_es, extrapolate=False)

#         fes = make_fes(np.min(uavg), np.min(f_es), cs)

#         fcomps.append({'fes': fes, 'fcontact': fcontact})

#     return fcomps


def make_p_es_func(xmin, pmin, cs):
    def _p_es(x, v):

        if x < xmin:
            return pmin * v**2
        else:
            return cs(x) * v**2

    return _p_es


def make_xmax_func(xmin, xmaxmin, cs):
    def _xmax(x):

        if x < xmin:
            return xmaxmin
        else:
            return cs(x)

    return np.vectorize(_xmax)


def make_p_cont_spr_func(k, n, x0):
    def _p_cont_spr(x):

        if x >= x0:
            return 0
        else:
            return k * (x0 - x)**n

    return _p_cont_spr


def make_p_cont_dmp_func(lmbd, n, x0):
    def _p_cont_dmp(x, xdot):

        if x >= x0:
            return 0
        else:
            return -(lmbd * (x0 - x)**n) * xdot

    return _p_cont_dmp


@util.memoize
def mem_patch_comp_funcs(mem, refn, lmbd, k, n, x0=None):
    '''
    Create compensation functions for each patch.
    '''
    f = fem.mem_patch_f_matrix(mem, refn)
    avg = fem.mem_patch_avg_matrix(mem, refn)

    K = fem.mem_k_matrix(mem, refn)
    Kinv = fem.inv_block(K)

    g = mem.gap
    g_eff = mem.gap + mem.isolation / mem.permittivity

    xprof = Kinv.dot(-np.sum(f, axis=1)).squeeze()
    xprof = xprof / np.max(np.abs(xprof))

    comp_funcs = []

    for i, pat in enumerate(mem.patches):

        avg_pat = avg[:, i]
        xnorm = xprof / np.max(np.abs(xprof[avg_pat > 0]))

        if x0 is None:
            x0 = (xnorm * g).dot(avg_pat)

        p_es = []
        xavg = []
        xmax = []

        for i, d in enumerate(np.linspace(-2, 2, 81)):

            x = xnorm * g * d
            x[x < -g] = -g

            _xavg = x.dot(avg_pat)
            _xmax = np.abs(x[avg_pat > 0]).max() * np.sign(-d)
            _p_es = -e_0 / 2 / (x + g_eff)**2

            if i > 0:
                if _xavg == xavg[i - 1]:
                    break

            xavg.append(_xavg)
            xmax.append(_xmax)
            p_es.append(_p_es.dot(avg_pat))

        p_es = np.array(p_es)[::-1]
        xavg = np.array(xavg)[::-1]
        xmax = np.array(xmax)[::-1]

        cs_p_es = CubicSpline(xavg, p_es, extrapolate=False)
        p_es_func = make_p_es_func(np.min(xavg), np.min(p_es), cs_p_es)

        cs_xmax = CubicSpline(xavg, xmax, extrapolate=False)
        xmax_func = make_xmax_func(np.min(xavg), np.min(xmax), cs_xmax)

        p_cont_spr_func = make_p_cont_spr_func(k, n, x0)

        p_cont_dmp_func = make_p_cont_dmp_func(lmbd, n, x0)

        comp_funcs.append({
            'p_es': p_es_func,
            'xmax': xmax_func,
            'p_cont_spr': p_cont_spr_func,
            'p_cont_dmp': p_cont_dmp_func
        })

    return comp_funcs


@util.memoize
def mem_static_x_vector(mem, refn, vdc, k, x0, atol=1e-10, maxiter=100):
    '''
    '''
    def pes(v, x, g_eff):
        return -e_0 / 2 * v**2 / (g_eff + x)**2

    if isinstance(mem, abstract.SquareCmutMembrane):
        amesh = mesh.square(mem.length_x, mem.length_y, refn)
    else:
        amesh = mesh.circle(mem.radius, refn)

    K = fem.mem_k_matrix(mem, refn)
    g_eff = mem.gap + mem.isolation / mem.permittivity
    F = fem.mem_f_vector(mem, refn, 1)
    Kinv = np.linalg.inv(K)
    nnodes = K.shape[0]
    x = np.zeros(nnodes)

    for i in range(maxiter):
        p = F * pes(vdc, x, g_eff)
        p[x < x0] += -k * (x[x < x0] - x0) * F[x < x0]

        xnew = Kinv.dot(p)

        if np.max(np.abs(xnew - x)) < atol:
            is_collapsed = False
            return xnew, is_collapsed

        x = xnew

    is_collapsed = True
    return x, p, is_collapsed


@util.memoize
def mem_patch_comp_funcs2(mem, refn, lmbd, k, n, x0=None):
    '''
    Create compensation functions for each patch.
    '''
    f = fem.mem_patch_f_matrix(mem, refn)
    avg = fem.mem_patch_avg_matrix(mem, refn)

    K = fem.mem_k_matrix(mem, refn)
    Kinv = fem.inv_block(K)

    g = mem.gap
    g_eff = mem.gap + mem.isolation / mem.permittivity

    xprof = Kinv.dot(-np.sum(f, axis=1)).squeeze()
    xprof = xprof / np.max(np.abs(xprof))

    comp_funcs = []

    for i, pat in enumerate(mem.patches):

        avg_pat = avg[:, i]
        xnorm = xprof / np.max(np.abs(xprof[avg_pat > 0]))

        if x0 is None:
            x0 = (xnorm * g).dot(avg_pat)

        p_es = []
        xavg = []
        xmax = []

        for i, d in enumerate(np.linspace(-2, 2, 81)):

            x = xnorm * g * d
            x[x < -g] = -g

            _xavg = x.dot(avg_pat)
            _xmax = np.abs(x[avg_pat > 0]).max() * np.sign(-d)
            _p_es = -e_0 / 2 / (x + g_eff)**2

            if i > 0:
                if _xavg == xavg[i - 1]:
                    break

            xavg.append(_xavg)
            xmax.append(_xmax)
            p_es.append(_p_es.dot(avg_pat))

        p_es = np.array(p_es)[::-1]
        xavg = np.array(xavg)[::-1]
        xmax = np.array(xmax)[::-1]

        cs_p_es = CubicSpline(xavg, p_es, extrapolate=False)
        p_es_func = make_p_es_func(np.min(xavg), np.min(p_es), cs_p_es)

        cs_xmax = CubicSpline(xavg, xmax, extrapolate=False)
        xmax_func = make_xmax_func(np.min(xavg), np.min(xmax), cs_xmax)

        p_cont_spr_func = make_p_cont_spr_func(k, n, x0)

        p_cont_dmp_func = make_p_cont_dmp_func(lmbd, n, x0)

        comp_funcs.append({
            'p_es': p_es_func,
            'xmax': xmax_func,
            'p_cont_spr': p_cont_spr_func,
            'p_cont_dmp': p_cont_dmp_func
        })

    return comp_funcs


def array_patch_comp_funcs(array, refn, lmbd, k, n, **kwargs):
    '''
    '''
    fcomps = []
    for elem in array.elements:
        for mem in elem.membranes:
            fcomps += mem_patch_comp_funcs(mem, refn, lmbd, k, n, **kwargs)

    return fcomps
