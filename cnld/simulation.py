'''
'''
import numpy as np
import scipy as sp
import scipy.signal
from scipy.constants import epsilon_0 as e_0
from scipy.interpolate import interp1d

from cnld import util, fem, mesh, impulse_response


# @util.memoize
# def mem_collapse_voltage(mem, maxdc=100, atol=1e-10,  maxiter=100):
#     '''
#     '''
#     for i in range(1, maxdc):
#         _, is_collapsed = mem_static_disp(K, e_mask, i, h_eff, tol)
        
#         if is_collapsed:
#             return i
#     raise('Could not find collapse voltage')


# @util.memoize
# def mem_static_disp(mem, vdc, refn=7, atol=1e-10, maxiter=100):
#     '''
#     '''
#     mem_mesh = mesh.square(mem.length_x, mem.length_y, refn)
#     K = fem.mem_k_matrix(mem_mesh, mem.y_modulus, m.thickness, m.p_ratio)
#     g_eff = mem.gap + mem.isol / mem.permittivity
#     F = fem.mem_f_vector(mem_mesh, 1)

#     nnodes = K.shape[0]
#     x0 = np.zeros(nnodes)

#     for i in range(maxiter):
#         x0_new = Kinv.dot(F * pressure_es(vdc, x0, g_eff)).squeeze()
        
#         if np.max(np.abs(x0_new - x0)) < atol:
#             is_collapsed = False
#             return x0_new, is_collapsed
        
#         x0 = x0_new

#     is_collapsed = True
#     return x0, is_collapsed


def pressure_es(v, x, g_eff):
    '''
    '''
    return -e_0 / 2 * v**2 / (x + g_eff)**2


def convolve_fir(A, b, fs):
    '''
    '''
    nsrc, ndest, nfir = A.shape
    _, nsample = b.shape
    # x = np.zeros(ndest)

    if b.ndim == 1:
        b = b[...,None]

    x = np.sum(np.sum(A[:,:,:nsample] * b[:,None,::-1], axis=-1), axis=0).squeeze()
    # for i in range(nsrc):
        # for j in range(ndest):
            # x[i,j] += np.sum(A[i,j,:nsample] * b[i,::-1]) / fs
    return x


def gausspulse(fc, fbw, fs, tpr=-100, retquad=False):
    '''
    '''
    cutoff = sp.signal.gausspulse('cutoff', fc=fc, bw=fbw, tpr=tpr, bwr=-3)
    adj_cutoff = np.ceil(cutoff * fs) / fs

    t = np.arange(-adj_cutoff, adj_cutoff + 1 / fs, 1 / fs)
    pulse, quad = sp.signal.gausspulse(t, fc=fc, bw=fbw, retquad=True, bwr=-3)
    
    if retquad:
        return quad
    
    return pulse





class FixedStepSolver:
    
    def __init__(self, fir_t, fir, v_t, v, x0, t_start, t_stop, rtol=1e-3):
        

        # fir_t, fir = impulse_response.read_db(file)
        min_step = fir_t[1] - fir_t[0]

        self.voltage = interp1d(v_t, v)

        p0 = pressure_es(self.voltage(t_start), x0, g_eff)

        self.fir = fir
        self.fir_t = fir_t

        self.time = [0,]
        self.displacement = [x0,]
        self.pressure = [p0,]
        self.delta = []

        self.min_step = min_step
        self.gap_eff = g_eff
        
    @classmethod
    def from_db(cls, array, dbfile, v_t, v, x0, t_start, t_stop, rtol=1e-3):
        pass

    def step_check(self, x_new=None, append=False):
        
        p = np.array(self.pressure).copy()
        t = self.time[-1]
        fir = self.fir
        fs = 1  / self.min_step
        x = self.displacement[-1]

        t_new = t + self.min_step
        if x_new is None:
            x_new = convolve_fir(fir, p, fs, offset=1)
        v_new = self.voltage(t_new)
        p_new = pressure_es(v_new, x_new, self.gap_eff)

        p = np.append(p, np.atleast_2d(p_new), axis=0)

        x_check = convolve_fir(fir, p, fs, offset=0)
        delta = np.max(np.abs(x_new - x_check)) / self.gap_eff
        if not append:
            self.step_check(x_new=x_check, append=True)

        if append:
            self.displacement.append(x_new)
            self.pressure.append(p_new)
            self.time.append(t_new)
            self.delta.append(delta)
    

class VariableStepSolver(FixedStepSolver):
    
    def stepn(self, n):
        pass


        
    

    
