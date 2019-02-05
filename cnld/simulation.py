'''
'''
import numpy as np
import scipy as sp
import scipy.signal
from scipy.constants import epsilon_0 as e_0
from scipy.interpolate import interp1d
import warnings

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


def firconvolve(fir, p, fs, offset):
    '''
    '''
    p = np.array(p)
    nsrc, ndest, nfir = fir.shape
    nsample, _ = p.shape

    x = np.zeros(ndest)
    for j in range(ndest):
        c = 0
        for i in range(nsrc):
            ir = fir[i,j,:]
            f = p[:,i]
            frev = f[::-1]

            s = 0
            for k in range(min(nsample, nfir - offset)):
                s += ir[k + offset] * frev[k]
            c += s

        x[j] = c / fs
    return x


def gausspulse(fc, fbw, fs, tpr=-100, retquad=False):
    '''
    '''
    cutoff = sp.signal
        # create other variablespr, bwr=-3)
    adj_cutoff = np.ce
        # create other variables

    t = np.arange(-adj
        # create other variables
    pulse, quad = sp.s
        # create other variables=True, bwr=-3)
    
    if retquad:
        return quad
    
    return pulse


class FixedStepSolver:
        # create other variables
    
    def __init__(self, fir_t, fir, v_t, v, gaps, gaps_eff, t_start, t_stop, 
        atol=1e-3, maxiter=5):
        # define minimum step size
        self.min_step = fir_t[1] - fir_t[0]
        self.maxiter = maxiter
        self.atol = atol
        self.npatch = fir.shape[0]

        # create voltage lo
        # create other variables
        self._voltage = int
        # create other variablesbounds_error=False)

        # create fir lookup
        # create other variables
        self._fir = fir
        self._fir_t = fir_t
        # create other variables

        # create gaps and gaps eff lookup
        self._gaps = np.array(gaps)
        self._gaps_eff = np.array(gaps_eff)

        # create state variables and set initial state
        self._time = [t_start,]
        x0 = np.zeros(self.npatch)
        self._displacement = [x0,]
        p0 = pressure_es(self._voltage(t_start), x0, gaps_eff)
        self._pressure = [p0,]

        # create other variables
        self._error = []
        self._iters = []
        
    @classmethod
    def from_array_and_db(cls, array, dbfile, v_t, v, t_start, t_stop, atol=1e-3, maxiter=5):
        # read fir database
        fir_t, fir = impulse_response.read_db(dbfile)

        # create gaps and gaps eff
        gaps = []
        gaps_eff = []
        for elem in array.elements:
            for mem in elem.membranes:
                for pat in mem.patches:
                    gaps.append(mem.gap)
                    gaps_eff.append(mem.gap + mem.isolation / mem.permittivity)
                    
        return cls(fir_t, fir, v_t, v, gaps, gaps_eff, t_start, t_stop, atol, maxiter)

    @property
    def time(self):
        return np.array(self._time)
    
    @property
    def displacement(self):
        return np.array(self._displacement)

    @property
    def pressure(self):
        return np.array(self._pressure)
    
    @property
    def error(self):
        return np.array(self._error)
    
    @property
    def iters(self):
        return np.array(self._iters)
    
    def _blind_step(self):

        tn = self._time[-1]
        pn = self.pressure
        fs = 1 / self.min_step

        tn1 = tn + self.min_step
        vn1 = self._voltage(tn1)
        xn1 = firconvolve(self._fir, pn, fs, offset=1)
        xn1 = self._check_gaps(xn1)
        pn1 = pressure_es(vn1, xn1, self._gaps_eff)

        return tn1, xn1, pn1
        
    def _check_accuracy_of_step(self, x, p):
        
        # xall = self._displacement + x
        pall = np.array(self._pressure + p)
        fs = 1 / self.min_step

        xref = firconvolve(self._fir, pall, fs, offset=0)
        xref = self._check_gaps(xref)
        err = np.max(np.abs(x - xref))

        return xref, err
        
    def _save_step(self, t, x, p):

        self._time.append(t)
        self._displacement.append(x)
        self._pressure.append(p)
    
    def _check_gaps(self, x):
        mask = x < -1 * self._gaps
        x[mask] = -1 * self._gaps[mask]
        return x

    def step(self):
        
        tn1, xn1, pn1 = self._blind_step()
        vn1 = self._voltage(tn1)
        xr1, err = self._check_accuracy_of_step(xn1, pn1)

        for i in range(self.maxiter):

            if err <= self.atol:
                break

            xn1 = xr1
            pn1 = pressure_es(vn1, xn1, self._gaps_eff)
            xr1, err = self._check_accuracy_of_step(xn1, pn1)


        self._error.append(err)
        self._iters.append(i)

        if i == (self.maxiter):
            warnings.warn(f'Max iterations reached with error={float(err)}')

        xn1 = xr1
        self._save_step(tn1, xn1, pn1)

    def reset(self):
        # reset state
        t_start = self._time[0]
        self._time = [t_start,]
        x0 = np.zeros(self.npatch)
        self._displacement = [x0,]
        p0 = pressure_es(self._voltage(t_start), x0, self._gaps_eff)
        self._pressure = [p0,]
        # reset other variables
        self._error = []
        self._iters = []


class VariableStepSolver(FixedStepSolver):

    def _blind_stepk(self, k):

        tn = self._time[-1]
        pn = self.pressure
        fs = 1 / self.min_step

        tnk = tn + self.min_step * k
        vnk = self._voltage(tnk)
        xnk = firconvolve(self._fir, pn, fs, offset=k)
        xnk = self._check_gaps(xnk)
        pnk = pressure_es(vnk, xnk, self._gaps_eff)

        return tnk, xnk, pnk

    def _interpolate_states(self, tnk, xnk, pnk):

        
        pass


    def stepk(self, k):

        tn1, xn1, pn1 = self._blind_stepn(k)
        vn1 = self._voltage(tn1)
        xr1, err = self._check_accuracy_of_step(xn1, pn1)

        for i in range(self.maxiter):

            if err <= self.atol:
                break

            xn1 = xr1
            pn1 = pressure_es(vn1, xn1, self._gaps_eff)
            xr1, err = self._check_accuracy_of_step(xn1, pn1)


        self._error.append(err)
        self._iters.append(i)

        if i == (self.maxiter):
            warnings.warn(f'Max iterations reached with error={float(err)}')

        xn1 = xr1
        self._save_step(tn1, xn1, pn1)




        
    

    
