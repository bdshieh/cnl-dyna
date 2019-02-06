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


class VariableStepSolver(FixedStepSolver):

    def _blind_stepk(self, k):

        tn = self._time[-1]
        pn = self.pressure
        fs = 1 / self.min_step

        tnk = tn + self.min_step * k
        vnk = self._voltage(tnk)
        xnk = self._check_gaps(firconvolve(self._fir, pn, fs, offset=k))
        pnk = pressure_es(vnk, xnk, self._gaps_eff)

        return tnk, xnk, pnk

    def _interpolate_states(self, k, tnk, xnk, pnk):

        tn = self._time[-1]
        xn = self._displacement[-1]
        # if len(self._time) < 2:
            # tn_1 = tn - self.min_step
            # xn_1 = xn
        # else:
        tn_1 = self._time[-2]
        xn_1 = self._displacement[-2]

        # tn_2 = self._time[-3]
        # xn_2 = self._displacement[-3]

        # fxi = interp1d([tn_2, tn_1, tn, tnk], [xn_2, xn_1, xn, xnk], axis=0, kind='cubic')
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

    def _check_accuracy_of_step(self, x, p):
        
        pall = np.array(self._pressure + p)
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
        vnk = self._voltage(tnk)
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
    

class StepDoublingSolver(VariableStepSolver):

    def _blind_stepk(self, k, append_states=None):

        fs = 1 / self.min_step

        if append_states is None:
            tn, xn, pn = self._time[-1], self._displacement[-1], self._pressure[-1]
            tn_1, xn_1, pn_1 = self._time[-2], self._displacement[-2], self._pressure[-2]
            pall = np.array(self._pressure)
        else:
            ts, xs, ps = append_states

            tn, xn, pn = ts[-1], xs[-1], ps[-1]
            if len(ts) < 2:
                tn_1, xn_1, pn_1 = self._time[-1], self._displacement[-1], self._pressure[-1]
            else:
                tn_1, xn_1, pn_1 = ts[-2], xs[-2], ps[-2]
            pall = np.array(self._pressure + ps)

        tnk = tn + self.min_step * k
        xnk = self._check_gaps(firconvolve(self._fir, pall, fs, offset=k))
        vnk = self._voltage(tnk)
        pnk = pressure_es(vnk, xnk, self._gaps_eff)

        ts, xs, ps = self._interpolate_between_states(k, (tnk, xnk, pnk), (tn_1, xn_1, pn_1), (tn, xn, pn))

        return ts, xs, ps

    def _interpolate_between_states(self, k, statek, staten_1=None, staten=None):
        
        if staten_1 is None:
            tn_1, xn_1, pn_1 = self._time[-2], self._displacement[-2], self._pressure[-2]
        else:
            tn_1, xn_1, pn_1 = staten_1
        if staten is None:
            tn, xn, pn = self._time[-1], self._displacement[-1], self._pressure[-1]
        else:
            tn, xn, pn = staten

        tnk, xnk, pnk = statek

        fxi = interp1d([tn_1, tn, tnk], [xn_1, xn, xnk], axis=0, kind='quadratic')

        ts = []
        xs = []
        ps = []
        for i in range(k):
            tt = tn + self.min_step * (i + 1)

            xt = self._check_gaps(fxi(tt))
            vt = self._voltage(tt)
            pt = pressure_es(vt, xt, self._gaps_eff)

            ts.append(tt)
            xs.append(xt)
            ps.append(pt)
            
        return ts, xs, ps

    def _check_accuracy_of_steps(self, states):
        
        ts, xs, ps = states
        xnk = xs[-1]
        pall = np.array(self._pressure + ps)
        fs = 1 / self.min_step

        xrk = self._check_gaps(firconvolve(self._fir, pall, fs, offset=0))
        err = np.max(np.abs(xnk - xrk))

        return xrk, err

    def _save_steps(self, ts, xs, ps):

        self._time += ts
        self._displacement += xs
        self._pressure += ps

    def _determine_step_size(self):

        k = 6

        while True:

            states_max = self._blind_stepk(k)

            states_half = self._blind_stepk(k // 2)
            states_half = self._blind_stepk(k - (k // 2), append_states=states_half)

            xmax = states_max[1]
            xhalf = states_half[1]

            err = np.max(np.abs(xmax[-1] - xhalf[-1]))
            if err <= 10 * self.atol:
                break
            
            k = int(k - 1)

            if k <= 2:
                k = 1
                break
                
        return k

    def doublingstep(self):

        k = self._determine_step_size()
        print(k)

        ts, xs, ps = self._blind_stepk(k)
        tnk = ts[-1]
        vnk = self._voltage(tnk)

        xrk, err = self._check_accuracy_of_steps((ts, xs, ps))

        for i in range(self.maxiter):
            if err <= self.atol:
                break

            xnk = xrk
            pnk = pressure_es(vnk, xnk, self._gaps_eff)
            ts, xs, ps = self._interpolate_between_states(k, (tnk, xnk, pnk))
            xrk, err = self._check_accuracy_of_steps((ts, xs, ps))

        self._error.append(err)
        self._iters.append(i)

        if i == (self.maxiter):
            warnings.warn(f'Max iterations reached with error={float(err)}')

        self._save_steps(ts, xs, ps)



t_start = 0
t_stop = 10e-6
v_t = np.arange(0, 10e-6, 5e-9)
vdc = 1 / (1 + np.exp(-25e6 * (v_t - 2e-6)))
vac = np.sin(2 * np.pi * 1e6 * v_t)
v = 22 * vdc
atol = 1e-10
array = abstract.load('matrix.json')
dbfile = 'imp_resp.db'

solver = StepDoublingSolver.from_array_and_db(array, dbfile, v_t, v, t_start, t_stop, atol)
solver2 = FixedStepSolver.from_array_and_db(array, dbfile, v_t, v, t_start, t_stop, atol)

for i in range(600):
    solver2.step()

solver.step()
solver.step()
for i in range(200):
    solver.doublingstep()

# from scipy.signal import fftconvolve
# pr = (solver.pressure).T[:,None,:]
# xr = np.sum(fftconvolve(solver._fir, pr, axes=-1) * solver.min_step, axis=0)

fig, ax = plt.subplots()
ax.plot(solver.time / 1e-9, np.array(solver.displacement)[:,[0, 1, 4]] / 1e-9, 'o-', color='blue', fillstyle='none', markersize=2)
ax.plot(solver2.time / 1e-9, np.array(solver2.displacement)[:,[0, 1, 4]] / 1e-9, 'o-', color='red', fillstyle='none', markersize=2)
tax = ax.twinx()
tax.plot(v_t / 1e-9, v, '--', color='orange')
ax.set_ylabel('Displacement (nm)')
ax.set_xlabel('Time (ns)')
tax.set_ylabel('Voltage (V)')
fig.show()

# fig, ax = plt.subplots()
# ax.plot(solver.time, np.array(solver.pressure)[:,4] / 1e-9, '.-')
# tax = ax.twinx()
# tax.plot(v_t, v, '--', color='orange')
# ax.set_title('Pressure')
# fig.show()