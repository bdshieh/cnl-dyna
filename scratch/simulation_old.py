'''
'''
import numpy as np
import scipy as sp
import scipy.signal
from scipy.constants import epsilon_0 as e_0
from scipy.interpolate import interp1d
import warnings
import numpy.linalg

from cnld import abstract, fem, mesh, impulse_response, compensation, database


# @util.memoize
# def mem_collapse_voltage(mem, maxdc=100, atol=1e-10,  maxiter=100):
#     '''
#     '''
#     for i in range(1, maxdc):
#         _, is_collapsed = mem_static_disp(K, e_mask, i, h_eff, tol)
        
#         if is_collapsed:
#             return i
#     raise('Could not find collapse voltage')


def pressure_es2(v, x, fc):
    '''
    '''
    return [f(xi) * v**2 for f, xi in zip(fc, x)]


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


class CompensatingFixedStepSolver:
    
    def __init__(self, fir_t, fir, v_t, v, gaps, gaps_eff, t_start, t_stop, fcomp, atol=1e-10, maxiter=5):

        # define minimum step size
        self.min_step = fir_t[1] - fir_t[0]
        self.maxiter = maxiter
        self.atol = atol
        self.npatch = fir.shape[0]

        # create voltage look up function
        self._voltagef = interp1d(v_t, v, axis=-1, fill_value=0, bounds_error=False)

        # create fir lookup
        self._fir = fir
        self._fir_t = fir_t

        # create gaps and gaps eff lookup
        self._gaps = np.array(gaps)
        self._gaps_eff = np.array(gaps_eff)
        self._fcomp = fcomp

        # create state variables and set initial state
        self._time = [t_start,]
        x0 = np.zeros(self.npatch)
        self._displacement = [x0,]
        v0 = self._voltagef(t_start)
        self._voltage = [v0,]
        p0 = pressure_es2(self._voltagef(t_start), x0, fcomp)
        self._pressure = [p0,]

        # create other variables
        self._error = []
        self._iters = []
        self._t_stop = t_stop
        
    @classmethod
    def from_array_and_db(cls, array, refn, dbfile, v_t, v, t_start, t_stop, atol=1e-10, maxiter=5):

        # read fir database
        # fir_t, fir = impulse_response.read_db(dbfile)
        fir_t, fir = database.read_patch_to_patch_imp_resp(dbfile)

        # create gaps and gaps eff
        gaps = []
        gaps_eff = []
        for elem in array.elements:
            for mem in elem.membranes:
                for pat in mem.patches:
                    gaps.append(mem.gap)
                    gaps_eff.append(mem.gap + mem.isolation / mem.permittivity)
        
        # fcomp = compensation.fcomp_from_abstract(array, refn)
        fcomp = compensation.array_patch_fcomp_funcs(array, refn)
        
        return cls(fir_t, fir, v_t, v, gaps, gaps_eff, t_start, t_stop, fcomp, atol, maxiter)

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
    def voltage(self):
        return np.array(self._voltage)
    
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
        vn1 = self._voltagef(tn1)
        xn1 = self._check_gaps(firconvolve(self._fir, pn, fs, offset=1))
        pn1 = pressure_es2(vn1, xn1, self._fcomp)

        return tn1, xn1, pn1
        
    def _check_accuracy_of_step(self, x, p):
        
        pall = np.array(self._pressure + [p,])
        fs = 1 / self.min_step

        xref = self._check_gaps(firconvolve(self._fir, pall, fs, offset=0))
        err = np.max(np.abs(x - xref))

        return xref, err
        
    def _save_step(self, t, x, p, v):

        self._time.append(t)
        self._displacement.append(x)
        self._pressure.append(p)
        self._voltage.append(v)
    
    def _check_gaps(self, x):
        mask = x < -1 * self._gaps
        x[mask] = -1 * self._gaps[mask]
        return x

    def step(self):
        
        tn1, xn1, pn1 = self._blind_step()
        vn1 = self._voltagef(tn1)
        xr1, err = self._check_accuracy_of_step(xn1, pn1)

        i = 1
        for j in range(self.maxiter - 1):
            if err <= self.atol:
                break

            xn1 = xr1
            pn1 = pressure_es2(vn1, xn1, self._fcomp)
            xr1, err = self._check_accuracy_of_step(xn1, pn1)
            i += 1

        self._error.append(err)
        self._iters.append(i)

        xn1 = xr1
        pn1 = pressure_es2(vn1, xn1, self._fcomp)
        self._save_step(tn1, xn1, pn1, vn1)

    def solve(self):

        t_stop = self._t_stop
        while True:
            self.step()
            if self.time[-1] >= t_stop:
                break

    def reset(self):
        
        t_start = self._time[0]
        self._time = [t_start,]
        x0 = np.zeros(self.npatch)
        self._displacement = [x0,]
        p0 = pressure_es2(self._voltagef(t_start), x0, fcomp)
        self._pressure = [p0,]

        # create other variables
        self._error = []
        self._iters = []


class FixedStepSolver:
    
    # def __init__(self, fir_t, fir, v_t, v, gaps, gaps_eff, t_start, t_stop, fcomp, atol=1e-10, maxiter=5):
    def __init__(self, fir_t, fir, v_t, v, gaps, gaps_eff, t_start, t_stop, atol=1e-10, maxiter=5):
        # define minimum step size
        self.min_step = fir_t[1] - fir_t[0]
        self.maxiter = maxiter
        self.atol = atol
        self.npatch = fir.shape[0]

        # create voltage look up function
        self._voltagef = interp1d(v_t, v, axis=-1, fill_value=0, bounds_error=False)

        # create fir lookup
        self._fir = fir
        self._fir_t = fir_t

        # create gaps and gaps eff lookup
        self._gaps = np.array(gaps)
        self._gaps_eff = np.array(gaps_eff)
        # self._fcomp = fcomp

        # create state variables and set initial state
        self._time = [t_start,]
        x0 = np.zeros(self.npatch)
        self._displacement = [x0,]
        v0 = self._voltagef(t_start)
        self._voltage = [v0,]
        # p0 = pressure_es2(self._voltagef(t_start), x0, fcomp)
        p0 = pressure_es(self._voltagef(t_start), x0, gaps_eff)
        self._pressure = [p0,]

        # create other variables
        self._error = []
        self._iters = []
        self._t_stop = t_stop
        
    @classmethod
    def from_array_and_db(cls, array, dbfile, v_t, v, t_start, t_stop, atol=1e-10, maxiter=5):
        # read fir database
        fir_t, fir = database.read_patch_to_patch_imp_resp(dbfile)

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
    def voltage(self):
        return np.array(self._voltage)
    
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
        vn1 = self._voltagef(tn1)
        xn1 = self._check_gaps(firconvolve(self._fir, pn, fs, offset=1))
        pn1 = pressure_es(vn1, xn1, self._gaps_eff)

        return tn1, xn1, pn1
        
    def _check_accuracy_of_step(self, x, p):
        
        pall = np.array(self._pressure + [p,])
        fs = 1 / self.min_step

        xref = self._check_gaps(firconvolve(self._fir, pall, fs, offset=0))
        err = np.max(np.abs(x - xref))

        return xref, err
        
    def _save_step(self, t, x, p, v):

        self._time.append(t)
        self._displacement.append(x)
        self._pressure.append(p)
        self._voltage.append(v)
    
    def _check_gaps(self, x):
        mask = x < -1 * self._gaps
        x[mask] = -1 * self._gaps[mask]
        return x

    def step(self):
        
        tn1, xn1, pn1 = self._blind_step()
        vn1 = self._voltagef(tn1)
        xr1, err = self._check_accuracy_of_step(xn1, pn1)

        i = 1
        for j in range(self.maxiter - 1):
            if err <= self.atol:
                break

            xn1 = xr1
            pn1 = pressure_es(vn1, xn1, self._gaps_eff)
            xr1, err = self._check_accuracy_of_step(xn1, pn1)
            i += 1

        self._error.append(err)
        self._iters.append(i)

        xn1 = xr1
        pn1 = pressure_es(vn1, xn1, self._gaps_eff)
        self._save_step(tn1, xn1, pn1, vn1)

    def solve(self):

        t_stop = self._t_stop
        while True:
            self.step()
            if self.time[-1] >= t_stop:
                break

    def reset(self):
        
        t_start = self._time[0]
        self._time = [t_start,]
        x0 = np.zeros(self.npatch)
        self._displacement = [x0,]
        p0 = pressure_es(self._voltagef(t_start), x0, self._gaps_eff)
        # p0 = pressure_es2(self._voltagef(t_start), x0, fcomp)
        self._pressure = [p0,]

        # create other variables
        self._error = []
        self._iters = []


class VariableStepSolver(FixedStepSolver):

    def _blind_stepk(self, k):

        tn = self._time[-1]
        pn = self.pressure
        fs = 1 / self.min_step

        tnk = tn + self.min_step * k
        vnk = self._voltagef(tnk)
        xnk = self._check_gaps(firconvolve(self._fir, pn, fs, offset=k))
        pnk = pressure_es(vnk, xnk, self._gaps_eff)

        return tnk, xnk, pnk

    def _interpolate_states(self, k, tnk, xnk, pnk):

        tn = self._time[-1]
        xn = self._displacement[-1]
        if len(self._time) < 2:
            tn_1 = tn - self.min_step
            xn_1 = xn
        else:
            tn_1 = self._time[-2]
            xn_1 = self._displacement[-2]
        
        fxi = interp1d([tn_1, tn, tnk], [xn_1, xn, xnk], axis=0, kind='cubic')
        
        t = []
        x = []
        p = []
        for i in range(k):
            tt = tn + self.min_step * (i + 1)

            xt = self._check_gaps(fxi(tt))
            vt = self._voltagef(tt)
            pt = pressure_es(vt, xt, self._gaps_eff)

            t.append(tt)
            x.append(xt)
            p.append(pt)
            
        return t, x, p

    def _check_accuracy_of_step(self, x, p):
        
        pall = np.array(self._pressure + [p,])
        fs = 1 / self.min_step

        xref = self._check_gaps(firconvolve(self._fir, pall, fs, offset=0))
        err = np.max(np.abs(x - xref))

        return xref, err

    def _save_steps(self, ts, xs, ps):

        self._time += ts
        self._displacement += xs
        self._pressure += ps

    def stepk(self, k):

        tnk, xnk, pnk = self._blind_stepk(k)
        vnk = self._voltagef(tnk)
        ts, xs, ps = self._interpolate_states(k, tnk, xnk, pnk)
        xrk, err = self._check_accuracy_of_step(xs, ps)

        for i in range(self.maxiter):
            if err <= self.atol:
                break

            xnk = xrk
            pnk = pressure_es(vnk, xnk, self._gaps_eff)
            ts, xs, ps = self._interpolate_states(k, tnk, xnk, pnk)
            xrk, err = self._check_accuracy_of_step(xs, ps)

        self._error.append(err)
        self._iters.append(i)

        if i == (self.maxiter):
            warnings.warn(f'Max iterations reached with error={float(err)}')

        self._save_steps(ts, xs, ps)


def gaussian_pulse(fc, fbw, fs, td=0, tpr=-60, antisym=True):
    '''
    Gaussian pulse.
    '''
    cutoff = scipy.signal.gausspulse('cutoff', fc=fc, bw=fbw, tpr=tpr, bwr=-3)
    adj_cutoff = np.ceil(cutoff * fs) / fs

    t = np.arange(-adj_cutoff, adj_cutoff + 1 / fs / 2, 1 / fs)
    pulse, quad = sp.signal.gausspulse(t, fc=fc, bw=fbw, retquad=True, bwr=-3)

    t += td
    if antisym:
        return t, quad / np.max(quad)
    else:
        return t, pulse / np.max(pulse)


def logistic_ramp(tr, dt, td=0, tstop=None, tpr=-60):
    '''
    DC ramp defined by rise time using the logistic function.
    '''
    k = 2 * np.log(10**(-tpr / 20)) / tr
    cutoff = np.ceil(tr / 2 / dt) * dt
    if tstop is None:
        t = np.arange(-cutoff, cutoff + dt / 2, dt)
    else:
        t = np.arange(-cutoff, tstop - td + dt / 2, dt)

    v = 1 / (1 + np.exp(-k * t))
    t += td
    return t, v


def linear_ramp(tr, dt, td=0, tstop=None):
    '''
    DC linear ramp.
    '''
    def f(t):
        if t > tr:
            return 1
        else:
            return 1 / tr * t
    fv = np.vectorize(f)

    cutoff = np.ceil(tr / dt) * dt
    if tstop is None:
        t = np.arange(0, cutoff, dt)
    else:
        t = np.arange(0, tstop - td + dt / 2, dt)

    v = fv(t)
    t += td
    return t, v


def winsin(f, ncycle, dt, td=0):
    '''
    Windowed sine.
    '''
    cutoff = round(ncycle * 1 / f / 2 / dt) * dt
    t = np.arange(-cutoff, cutoff + dt / 2, dt)

    v = np.sin(2 * np.pi * f * t)
    v[0] = 0
    v[-1] = 0
    t += td
    return t, v


def sigadd(*args):
    '''
    Add multiple time signals together, zero-padding when necessary.
    '''
    t0 = [t[0] for t, v in args]
    tmin = min(t0)

    t = args[0][0]
    dt = t[1] - t[0]

    frontpad = [int(round((_t0 - tmin) / dt)) for _t0 in t0]
    maxlen = max([fpad + len(v) for fpad, (_, v) in zip(frontpad, args)])
    backpad = [maxlen - (fpad + len(v)) for fpad, (_, v) in zip(frontpad, args)]

    tnew = np.arange(0, maxlen) * dt + tmin
    vnew = np.zeros(len(tnew))
    for fpad, bpad, (t, v) in zip(frontpad, backpad, args):
        vnew += np.pad(v, ((fpad, bpad)), mode='edge')
    
    return tnew, vnew





        
    

    