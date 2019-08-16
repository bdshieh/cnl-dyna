'''
'''
import numpy as np
cimport numpy as np
import cython
cimport cython
import scipy as sp
import scipy.signal
from scipy.constants import epsilon_0 as e_0
from scipy.interpolate import interp1d
from namedlist import namedlist

from cnld import abstract, impulse_response, compensation, database, fem, compensation


def electrostat_pres(v, x, g_eff):
    '''
    '''
    return -e_0 / 2 * v**2 / (x + g_eff)**2


def fir_conv_py(fir, p, fs, offset):
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


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray fir_conv_cy(double[:,:,:] fir, double[:,:] p, double dt, int offset):
    '''
    '''
    cdef int nsrc = fir.shape[0]
    cdef int ndest = fir.shape[1]
    cdef int nfir = fir.shape[2]
    cdef int nsample = p.shape[0]
    cdef double[:] x = np.zeros(ndest, dtype=np.float64)
    cdef double c, s
    cdef int i, j, k

    for j in range(ndest):

        c = 0
        for i in range(nsrc):

            s = 0
            for k in range(min(nsample, nfir - offset)):
                s += fir[i, j, k + offset] * p[nsample - 1 - k, i] 

            c += s

        x[j] = c * dt

    return np.asarray(x)


def make_f_cont_spr(k, n, x0):
    def _f_cont_spr(x):
        if x >= x0:
            return 0
        return k * (x0 - x)**n
    return np.vectorize(_f_cont_spr)


def make_f_cont_dmp(lmbd, n, x0):
    def _f_cont_dmp(x, xdot):
        if x >= x0:
            return 0
        return -(lmbd * (x0 - x)**n) * xdot 
    return np.vectorize(_f_cont_dmp)


class FixedStepSolver:
    
    State = namedlist('State', 
    '''
    displacement velocity voltage p_total p_es p_cont_spr p_cont_dmp  
    ''')
    Properties = namedlist('Properties', 'gap gap_effective')

    def __init__(self, t_fir, t_v, gap, gap_eff, t_lim, k, n, x0, lmbd, atol=1e-10, maxiter=5):

        fir_t, fir = t_fir
        v_t, v = t_v
        t_start, t_stop, t_step = t_lim

        npatch = fir.shape[0]
        
        # define time array
        time = np.arange(t_start, t_stop + t_step, t_step)
        ntime = len(time)
        
        # define voltage array (with interpolation)
        if v.ndim <= 1:
            v = np.tile(v, (npatch, 1)).T
        fi_voltage = interp1d(v_t, v, axis=0, fill_value=(v[0,:], v[-1,:]), bounds_error=False, kind='linear', assume_sorted=True)
        voltage = fi_voltage(time)

        # define fir (with interpolation)
        fi_fir = interp1d(fir_t, fir, axis=-1, kind='cubic', assume_sorted=True)
        self._fir_t = np.arange(fir_t[0], fir_t[-1], t_step)
        self._fir = fi_fir(self._fir_t)

        # define patch properties
        self._gap = np.array(gap)
        self._gap_eff = np.array(gap_eff)
        vmax = voltage.max(axis=0)

        # define patch state
        self._time = time
        self._voltage = voltage
        self._displacement = np.zeros((ntime, npatch))
        self._velocity = np.zeros((ntime, npatch))
        self._p_es = np.zeros((ntime, npatch))
        self._p_cont_spr = np.zeros((ntime, npatch))
        self._p_cont_dmp = np.zeros((ntime, npatch))
        self._p_total = np.zeros((ntime, npatch))

        # create other variables
        self._error = []
        self._iters = []
        self.atol = atol
        self.maxiter = maxiter
        self.npatch = npatch
        self.ntime = ntime
        self.current_step = 0
        self.min_step = t_step

        self._f_cont_spr = make_f_cont_spr(k, n, x0)
        self._f_cont_dmp = make_f_cont_dmp(lmbd, n, x0)

        # set initial state
        self._update_p_es(self.state_last, self.properties)
        self._update_p_cont_spr(self.state_last)
        self._update_p_cont_dmp(self.state_last)
        self._update_p_total(self.state_last) 

    @classmethod
    def from_array_and_db(cls, array, dbfile, t_v, t_lim, k, n, x0, lmbd, atol=1e-10, maxiter=5):
        # read fir database
        fir_t, fir = database.read_patch_to_patch_imp_resp(dbfile)

        # create gap and gap eff
        gap = []
        gap_eff = []
        for elem in array.elements:
            for mem in elem.membranes:
                for pat in mem.patches:
                    gap.append(mem.gap)
                    gap_eff.append(mem.gap + mem.isolation / mem.permittivity)
                
        return cls((fir_t, fir), t_v, gap, gap_eff, t_lim, k, n, x0, lmbd, atol, maxiter)

    @property
    def time(self):
        return np.array(self._time[:self.current_step + 1])

    @property
    def voltage(self):
        return np.array(self._voltage[:self.current_step + 1,:])
    
    @property
    def displacement(self):
        return np.array(self._displacement[:self.current_step + 1,:])
    
    @property
    def velocity(self):
        return np.array(self._velocity[:self.current_step + 1,:])

    @property
    def p_es(self):
        return np.array(self._p_es[:self.current_step + 1,:])

    @property
    def p_cont_spr(self):
        return np.array(self._p_cont_spr[:self.current_step + 1,:])

    @property
    def p_cont_dmp(self):
        return np.array(self._p_cont_dmp[:self.current_step + 1,:])

    @property
    def p_total(self):
        return np.array(self._p_total[:self.current_step + 1,:])

    @property
    def state(self):
        displacement = self.displacement
        velocity = self.velocity
        voltage = self.voltage
        p_total = self.p_total
        p_es = self.p_es
        p_cont_spr = self.p_cont_spr
        p_cont_dmp = self.p_cont_dmp
        
        return self.State(displacement=displacement, velocity=velocity, voltage=voltage,
            p_total=p_total, p_es=p_es, p_cont_spr=p_cont_spr, p_cont_dmp=p_cont_dmp)

    @property
    def state_previous(self):
        idx = self.current_step - 1
        if idx < 0:
            return None

        displacement = self._displacement[idx,:]
        velocity = self._velocity[idx,:]
        voltage = self._voltage[idx,:]
        p_total = self._p_total[idx,:]
        p_es = self._p_es[idx,:]
        p_cont_spr = self._p_cont_spr[idx,:]
        p_cont_dmp = self.p_cont_dmp[idx,:]

        return self.State(displacement=displacement, velocity=velocity, voltage=voltage,
            p_total=p_total, p_es=p_es, p_cont_spr=p_cont_spr, p_cont_dmp=p_cont_dmp)

    @property
    def state_last(self):
        idx = self.current_step
        displacement = self._displacement[idx,:]
        velocity = self._velocity[idx,:]
        voltage = self._voltage[idx,:]
        p_total = self._p_total[idx,:]
        p_es = self._p_es[idx,:]
        p_cont_spr = self._p_cont_spr[idx,:]
        p_cont_dmp = self.p_cont_dmp[idx,:]

        return self.State(displacement=displacement, velocity=velocity, voltage=voltage,
            p_total=p_total, p_es=p_es, p_cont_spr=p_cont_spr, p_cont_dmp=p_cont_dmp)
    
    @property
    def state_next(self):
        idx = self.current_step + 1
        displacement = self._displacement[idx,:]
        velocity = self._velocity[idx,:]
        voltage = self._voltage[idx,:]
        p_total = self._p_total[idx,:]
        p_es = self._p_es[idx,:]
        p_cont_spr = self._p_cont_spr[idx,:]
        p_cont_dmp = self._p_cont_dmp[idx,:]

        return self.State(displacement=displacement, velocity=velocity, voltage=voltage,
            p_total=p_total, p_es=p_es, p_cont_spr=p_cont_spr, p_cont_dmp=p_cont_dmp)

    @property
    def properties(self):
        return self.Properties(gap=self._gap, gap_effective=self._gap_eff)

    def _fir_conv(self, p, offset):
        return fir_conv_cy(self._fir, p, self.min_step, offset=offset)

    def _fir_conv_base(self, p):
        return fir_conv_cy(self._fir, p[:-1,:], self.min_step, offset=1)

    def _fir_conv_add(self, p):
        return fir_conv_cy(self._fir, p[-1:,:], self.min_step, offset=0)

    def _update_p_es(self, state, props):
        state.p_es[:] = electrostat_pres(state.voltage, state.displacement, props.gap_effective)

    def _update_p_cont_spr(self, state):
        state.p_cont_spr[:] = self._f_cont_spr(state.displacement)

    def _update_p_cont_dmp(self, state):
        state.p_cont_dmp[:] = self._f_cont_dmp(state.displacement, state.velocity)

    def _update_p_total(self, state):
        state.p_total[:] = state.p_cont_spr + state.p_cont_dmp + state.p_es

    def _update_velocity(self, state_last, state_next):
        state_next.velocity[:] = (state_next.displacement - state_last.displacement) / self.min_step

    def _blind_step(self):

        state = self.state
        state_last = self.state_last
        state_next = self.state_next
        props = self.properties

        state_next.displacement[:] = self._fir_conv(state.p_total, offset=1)
        self._update_velocity(state_last, state_next)

        self._update_p_es(state_next, props)
        self._update_p_cont_spr(state_next)
        # self._update_p_cont_dmp(state_next)
        self._update_p_total(state_next)
 
    def _check_accuracy_of_step(self, base=None):
        
        state_next = self.state_next

        p = self._p_total[:self.current_step + 2,:]

        if base is None:
            base = self._fir_conv_base(p)

        add = self._fir_conv_add(p)
        xr = base + add
        err = np.max(np.abs(state_next.displacement - xr))

        return xr, err, base
        
    def step(self):

        state_last = self.state_last
        state_next = self.state_next
        props = self.properties

        self._blind_step()
        xr, err, base = self._check_accuracy_of_step()

        i = 1
        for j in range(self.maxiter - 1):
            if err <= self.atol:
                break

            state_next.displacement[:] = xr
            self._update_velocity(state_last, state_next)

            self._update_p_es(state_next, props)
            self._update_p_cont_spr(state_next)
            # self._update_p_cont_dmp(state_next)
            self._update_p_total(state_next)
            
            xr, err, base = self._check_accuracy_of_step(base=base)
            i += 1

        # self._error.append(err)
        self._iters.append(i)

        state_next.displacement[:] = xr
        # x0 = xr.copy()
        self._update_velocity(state_last, state_next)

        self._update_p_es(state_next, props)
        self._update_p_cont_spr(state_next)
        self._update_p_cont_dmp(state_next)
        self._update_p_total(state_next)

        xr, err, base = self._check_accuracy_of_step(base=base)
        state_next.displacement[:] = xr
        self._update_velocity(state_last, state_next)
        # self._update_p_cont_dmp(state_next)
        # self._error.append(err)

        ##
        # if np.any(state_next.p_cont_dmp > 0):
        #     xr, err = self._check_accuracy_of_step()
        #     self._error.append(err)

        # if np.any(state_next.p_cont_dmp > 0):
        for j in range(10):

            state_next.displacement[:] = xr
            self._update_velocity(state_last, state_next)

            self._update_p_es(state_next, props)
            self._update_p_cont_spr(state_next)
    #         # self._update_p_cont_dmp(state_next)
            self._update_p_total(state_next)
            
            xr, err, base = self._check_accuracy_of_step(base=base)
            if err <= self.atol:
                break

        state_next.displacement[:] = xr
        self._update_velocity(state_last, state_next)
        self._error.append(err)
        ##

        self.current_step += 1
    
    def solve(self):

        stop = len(self._time) - 1
        while True:
            self.step()

            if self.current_step >= stop:
                break

    def reset(self):
        
        npatch = self.npatch
        ntime = self.ntime

        # reset state
        self._displacement = np.zeros((ntime, npatch))
        self._velocity = np.zeros((ntime, npatch))
        self._p_es = np.zeros((ntime, npatch))
        self._p_cont_spr = np.zeros((ntime, npatch))
        self._p_cont_dmp = np.zeros((ntime, npatch))
        self._p_total = np.zeros((ntime, npatch))

        # set initial state
        self._update_p_total(self.state_last) 
        self.current_step = 1

        # create other variables
        self._error = []
        self._iters = []

    def __iter__(self):
        return self

    def __next__(self):

        if self.current_step >= len(self._time) - 1:
            raise StopIteration
        
        self.step()

        return self.current_step


class CompensationSolver(FixedStepSolver):

    def __init__(self, t_fir, t_v, gap, gap_eff, t_lim, fcomps, atol=1e-10, maxiter=5):

        self._fcomps = fcomps

        super().__init__(t_fir, t_v, gap, gap_eff, t_lim, atol=atol, maxiter=maxiter)

    @classmethod
    def from_array_and_db(cls, array, refn, dbfile, t_v, t_lim, atol=1e-10, maxiter=10, **kwargs):
        # read fir database
        fir_t, fir = database.read_patch_to_patch_imp_resp(dbfile)

        # create gap and gap eff
        gap = []
        gap_eff = []
        for elem in array.elements:
            for mem in elem.membranes:
                for pat in mem.patches:
                    gap.append(mem.gap)
                    gap_eff.append(mem.gap + mem.isolation / mem.permittivity)

        fcomps = compensation.array_patch_fcomp_funcs(array, refn, **kwargs)

        return cls((fir_t, fir), t_v, gap, gap_eff, t_lim, fcomps, atol, maxiter)

    def _update_p_es(self, state, props):
        state.p_es[:] = electrostat_pres(state.voltage, state.displacement, props.gap_effective)


class StateDB:
    
    State = namedlist('State', 't x u v p_tot p_es p_cont_spr p_cont_dmp')

    def __init__(self, t_lim, t_v, npatch):

        v_t, v = t_v
        t_start, t_stop, t_step = t_lim
        
        # define time array
        t = np.arange(t_start, t_stop + t_step, t_step)
        nt = len(t)
        
        # define voltage array (with interpolation)
        if v.ndim <= 1:
            v = np.tile(v, (npatch, 1)).T
        fi_voltage = interp1d(v_t, v, axis=0, fill_value=(v[0,:], v[-1,:]), bounds_error=False, kind='linear', assume_sorted=True)
        voltage = fi_voltage(t)

        self._t = t
        self._v = voltage
        self._x = np.zeros((nt, npatch))
        self._u = np.zeros((nt, npatch))
        self._p_es = np.zeros((nt, npatch))
        self._p_cont_spr = np.zeros((nt, npatch))
        self._p_cont_dmp = np.zeros((nt, npatch))
        self._p_tot = np.zeros((nt, npatch))

    def get_state_i(self, i):
        return State(t=self._t[i], v=self._v[i, :], x=self._x[i, :], u=self._u[i, :], p_es=self._p_es[i, :],
                     p_cont_spr=self._p_cont_spr[i, :], p_cont_dmp=self._p_cont_dmp[i, :], 
                     p_tot=self._p_tot[i, :])
    
    def set_state_i(self, i, s):

        self._x[i, :] = s.x
        self._u[i, :] = s.u
        self._p_es[i, :] = s.p_es
        self._p_cont_spr[i, :] = s.p_cont_spr
        self._p_cont_dmp[i, :] = s.p_cont_dmp
        self._p_tot[i, :] = s.p_tot
    
    def get_state_t(self, t):

        i = int(np.min(np.abs(t - self._t)))
        return self.get_state_i(i)
    
    def set_state_t(self, t):

        i = int(np.min(np.abs(t - self._t)))
        self.set_state_i(i)
    

# A numerical model for CMUT contact dynamics 
# A scalable numerical model for CMUT non-linear dynamics and contact mechanics

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





        
    

    