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
        self._voltage = interp1d(v_t, v, axis=-1, fill_value=0, bounds_error=False)

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
        xn1 = firconvolve(self._fir, pn, fs, offset=1)
        xn1 = self._check_gaps(xn1)
        pn1 = pressure_es(vn1, xn1, self._gaps_eff)

        return tn1, xn1, pn1
        
    def _check_accuracy_of_step(self, x, p):
        
        pall = np.array(self._pressure + [p,])
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

        self._time = [self._time[0],]
        x0 = np.zeros(self.npatch)
        self._displacement = [x0,]
        p0 = pressure_es(self._voltage(t_start), x0, self._gaps_eff)
        self._pressure = [p0,]

        # create other variables
        self._error = []
        self._iters = []
    


t_start = 0
t_stop = 10e-6
v_t = np.arange(0, 10e-6, 5e-9)
vdc = 1 / (1 + np.exp(-25e6 * (v_t - 2e-6)))
vac = np.sin(2 * np.pi * 1e6 * v_t)
v = 22 * vdc
atol = 1e-10
array = abstract.load('matrix.json')
dbfile = 'imp_resp.db'

solver = FixedStepSolver.from_array_and_db(array, dbfile, v_t, v, t_start, t_stop, atol, maxiter=4)
# solver2 = FixedStepSolver.from_array_and_db(array, dbfile, v_t, v, t_start, t_stop, atol, maxiter=0)

for i in range(600):
    solver.step()
    # solver2.step()

from scipy.signal import fftconvolve

pr = (solver.pressure).T[:,None,:]
xr = np.sum(fftconvolve(solver._fir, pr, axes=-1) * solver.min_step, axis=0)

fig, ax = plt.subplots()
ax.plot(solver.time / 1e-9, np.array(solver.displacement)[:,[0, 1, 4]] / 1e-9, 'o-', fillstyle='none', markersize=2)
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
