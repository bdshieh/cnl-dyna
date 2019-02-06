'''
'''
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from tqdm import tqdm
from scipy.constants import epsilon_0 as e_0
from scipy.interpolate import interp1d
import warnings

from cnld import abstract, impulse_response


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


class FixedStepSolver:
    
    def __init__(self, fir_t, fir, v_t, v, gaps, gaps_eff, t_start, t_stop, 
        atol=1e-3, maxiter=5):
        # define minimum step size
        self.min_step = fir_t[1] - fir_t[0]
        self.maxiter = maxiter
        self.atol = atol
        self.npatch = fir.shape[0]

        # create voltage look up function
        self._voltage = interp1d(v_t, v, axis=-1, fill_value=(v[0], v[-1]), bounds_error=False)

        # create fir lookup
        self._fir = fir
        self._fir_t = fir_t

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
        xn1 = self._check_gaps(firconvolve(self._fir, pn, fs, offset=1))
        pn1 = pressure_es(vn1, xn1, self._gaps_eff)

        return tn1, xn1, pn1
        
    def _check_accuracy_of_step(self, x, p):
        
        pall = np.array(self._pressure + [p,])
        fs = 1 / self.min_step

        xref = self._check_gaps(firconvolve(self._fir, pall, fs, offset=0))
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
        
        t_start = self._time[0]
        self._time = [t_start,]
        x0 = np.zeros(self.npatch)
        self._displacement = [x0,]
        p0 = pressure_es(self._voltage(t_start), x0, self._gaps_eff)
        self._pressure = [p0,]

        # create other variables
        self._error = []
        self._iters = []


class VariableStepSolver:

    def __init__(self, fir_t, fir, v_t, v, gaps, gaps_eff, t_start, t_stop, 
        atol=1e-3, maxiter=5):
        # define minimum step size
        self.min_step = fir_t[1] - fir_t[0]
        self.maxiter = maxiter
        self.atol = atol
        self.npatch = fir.shape[0]

        # create voltage look up function
        self._voltage = interp1d(v_t, v, axis=-1, fill_value=(v[0], v[-1]), bounds_error=False)

        # create fir lookup
        self._fir = fir
        self._fir_t = fir_t

        # create gaps and gaps eff lookup
        self._gaps = np.array(gaps)
        self._gaps_eff = np.array(gaps_eff)

        # create state variables and set initial state
        self._time = [t_start,]
        x0 = np.zeros(self.npatch)
        self._displacement = [x0,]
        p0 = pressure_es(self._voltage(t_start), x0, gaps_eff)
        self._pressure = [p0,]

        self._time_t = []
        self._displacement_t = []
        self._pressure_t = []

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
    def time_t(self):
        return self._time + self._time_t

    @property
    def displacement_t(self):
        return self._displacement + self._displacement_t

    @property
    def pressure_t(self):
        return self._pressure + self._pressure_t

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

    def _reset_temp(self):
        self._time_t = []
        self._displacement_t = []
        self._pressure_t = []

    def _save_temp(self):
        self._time += self._time_t
        self._displacement += self._displacement_t
        self._pressure += self._pressure_t
        self._reset_temp()

    def _update_temp(self, ts, xs, ps):
        self._time_t += ts
        self._displacement_t += xs
        self._pressure_t += ps
    
    def _clear_temp(self, k):
        for i in range(k):
            self._time_t.pop()
            self._displacement_t.pop()
            self._pressure_t.pop()

    def _check_gaps(self, x):
        mask = x < -1 * self._gaps
        x[mask] = -1 * self._gaps[mask]
        return x

    def _blind_step_k(self, k):

        tn = self.time_t[-1]
        pn = np.array(self.pressure_t)
        fs = 1 / self.min_step

        tnk = tn + self.min_step * k
        vnk = self._voltage(tnk)
        xnk = self._check_gaps(firconvolve(self._fir, pn, fs, offset=k))
        pnk = pressure_es(vnk, xnk, self._gaps_eff)

        ts, xs, ps = self._interpolate_states(k, tnk, xnk, pnk)
        self._update_temp(ts, xs, ps)

    def _interpolate_states(self, k, tnk, xnk, pnk):

        tn = self.time_t[-1]
        xn = self.displacement_t[-1]

        tn_1 = self.time_t[-2]
        xn_1 = self.displacement_t[-2]

        fxi = interp1d([tn_1, tn, tnk], [xn_1, xn, xnk], axis=0, kind='quadratic')

        t = []
        x = []
        p = []
        for i in range(k):
            tt = tn + self.min_step * (i + 1)

            xt = self._check_gaps(fxi(tt))
            vt = self._voltage(tt)
            pt = pressure_es(vt, xt, self._gaps_eff)

            t.append(tt)
            x.append(xt)
            p.append(pt)
            
        return t, x, p

    def _check_accuracy(self):
        
        p = np.array(self.pressure_t)
        fs = 1 / self.min_step
        xn = self.displacement_t[-1]

        xr = self._check_gaps(firconvolve(self._fir, p, fs, offset=0))
        err = np.max(np.abs(xn - xr))

        return xr, err

    def step(self):

        tn = self.time_t[-1]
        pn = np.array(self.pressure_t)
        fs = 1 / self.min_step

        tnk = tn + self.min_step
        vnk = self._voltage(tnk)
        xnk = self._check_gaps(firconvolve(self._fir, pn, fs, offset=1))
        pnk = pressure_es(vnk, xnk, self._gaps_eff)

        self._update_temp([tnk,], [xnk,], [pnk,])
        self._save_temp()

    def _step_k(self, k, xnk):

        tn = self.time_t[-1]
        pn = np.array(self.pressure_t)
        fs = 1 / self.min_step

        tnk = tn + self.min_step * k
        vnk = self._voltage(tnk)
        pnk = pressure_es(vnk, xnk, self._gaps_eff)

        ts, xs, ps = self._interpolate_states(k, tnk, xnk, pnk)
        self._update_temp(ts, xs, ps)

    def step_k(self, k, save=False):

        self._blind_step_k(k)
        # vnk = self._voltage(tnk)
        xrk, err = self._check_accuracy()

        for i in range(self.maxiter):
            if err <= self.atol:
                break

            xnk = xrk
            self._clear_temp(k)
            self._step_k(k, xnk)
            xrk, err = self._check_accuracy()

        self._error.append(err)
        self._iters.append(i)

        if i == (self.maxiter):
            warnings.warn(f'Max iterations reached with error={float(err)}')

        if save:
            self._save_temp()
    
    def doubling_step(self):

        k = 10

        self.step_k(k)
        xmax = self._displacement_t[-1]

        self._reset_temp()
        self.step_k(k // 2)
        self.step_k(k - (k // 2))
        xhalf = self._displacement_t[-1]

        err = np.max(np.abs(xmax - xhalf))
        print(err)
        self._reset_temp()
        

t_start = 0
t_stop = 10e-6
v_t = np.arange(0, 10e-6, 5e-9)
vdc = 1 / (1 + np.exp(-25e6 * (v_t - 2e-6)))
vac = np.sin(2 * np.pi * 1e6 * v_t)
v = 22 * vdc
atol = 1e-10
array = abstract.load('matrix.json')
dbfile = 'imp_resp.db'

solver = VariableStepSolver.from_array_and_db(array, dbfile, v_t, v, t_start, t_stop, atol)
# solver2 = FixedStepSolver.from_array_and_db(array, dbfile, v_t, v, t_start, t_stop, atol)

# for i in range(600):
#     solver2.step()

solver.step()
solver.step()

for i in range(600):
    solver.step()
    solver.doubling_step()


# solver.step()
# for i in range(200):
#     solver.doublingstep()

# from scipy.signal import fftconvolve
# pr = (solver.pressure).T[:,None,:]
# xr = np.sum(fftconvolve(solver._fir, pr, axes=-1) * solver.min_step, axis=0)

# fig, ax = plt.subplots()
# ax.plot(solver.time / 1e-9, np.array(solver.displacement)[:,[0, 1, 4]] / 1e-9, 'o-', color='blue', fillstyle='none', markersize=2)
# ax.plot(solver2.time / 1e-9, np.array(solver2.displacement)[:,[0, 1, 4]] / 1e-9, 'o-', color='red', fillstyle='none', markersize=2)
# tax = ax.twinx()
# tax.plot(v_t / 1e-9, v, '--', color='orange')
# ax.set_ylabel('Displacement (nm)')
# ax.set_xlabel('Time (ns)')
# tax.set_ylabel('Voltage (V)')
# fig.show()

# fig, ax = plt.subplots()
# ax.plot(solver.time, np.array(solver.pressure)[:,4] / 1e-9, '.-')
# tax = ax.twinx()
# tax.plot(v_t, v, '--', color='orange')
# ax.set_title('Pressure')
# fig.show()
