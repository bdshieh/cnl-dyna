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


def p_es_pp(v, x, g_eff):
    '''
    Electrostatic pressure for parallel-plate capacitor.
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
cpdef np.ndarray fir_conv_cy(const double[:,:,:] fir, const double[:,:] p, double dt, int offset):
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


def make_p_cont_spr(k, n, x0):
    def _p_cont_spr(x):
        if x >= x0:
            return 0
        return k * (x0 - x)**n
    return np.vectorize(_p_cont_spr)


def make_p_cont_dmp(lmbd, n, x0):
    def _p_cont_dmp(x, xdot):
        if x >= x0:
            return 0
        return -(lmbd * (x0 - x)**n) * xdot 
    return np.vectorize(_p_cont_dmp)


class StateDB:
    
    State = namedlist('State', 'i t x u v p_tot p_es p_cont_spr p_cont_dmp', default=None)

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

        self._i = np.arange(nt)
        self._t = t
        self._v = voltage
        self._x = np.zeros((nt, npatch))
        self._u = np.zeros((nt, npatch))
        self._p_es = np.zeros((nt, npatch))
        self._p_cont_spr = np.zeros((nt, npatch))
        self._p_cont_dmp = np.zeros((nt, npatch))
        self._p_tot = np.zeros((nt, npatch))

    @property
    def i(self):
        arr = self._i.view()
        # arr.flags.writeable = False
        return arr

    @property
    def t(self):
        arr = self._t.view()
        # arr.flags.writeable = False
        return arr

    @property
    def v(self):
        arr = self._v.view()
        # arr.flags.writeable = False
        return arr

    @property
    def x(self):
        arr = self._x.view()
        # arr.flags.writeable = False
        return arr

    @property
    def u(self):
        arr = self._u.view()
        # arr.flags.writeable = False
        return arr

    @property
    def p_es(self):
        arr = self._p_es.view()
        # arr.flags.writeable = False
        return arr

    @property
    def p_cont_spr(self):
        arr = self._p_cont_spr.view()
        # arr.flags.writeable = False
        return arr

    @property
    def p_cont_dmp(self):
        arr = self._p_cont_dmp.view()
        # arr.flags.writeable = False
        return arr

    @property
    def p_tot(self):
        arr = self._p_tot.view()
        # arr.flags.writeable = False
        return arr

    def get_state_i(self, i):
        return self.State(i=self.i[i], t=self.t[i], v=self.v[i, :], x=self.x[i, :], u=self.u[i, :], p_es=self.p_es[i, :],
                          p_cont_spr=self.p_cont_spr[i, :], p_cont_dmp=self.p_cont_dmp[i, :], 
                          p_tot=self.p_tot[i, :])

    def get_state_t(self, t):

        i = int(np.argmin(np.abs(t - self.t)))
        return self.get_state_i(i)

    def set_state_i(self, s):

        i = s.i

        if s.x is not None: self._x[i, :] = s.x
        if s.u is not None: self._u[i, :] = s.u
        if s.p_es is not None: self._p_es[i, :] = s.p_es
        if s.p_cont_spr is not None: self._p_cont_spr[i, :] = s.p_cont_spr
        if s.p_cont_dmp is not None: self._p_cont_dmp[i, :] = s.p_cont_dmp
        if s.p_tot is not None: self._p_tot[i, :] = s.p_tot
    
    def set_state_t(self, s):

        t = s.t
        i = int(np.min(np.abs(t - self._t)))
        self.set_state_i(i)
    
    def clear(self):

        nt, npatch = self._x.shape
        
        self._x = np.zeros((nt, npatch))
        self._u = np.zeros((nt, npatch))
        self._p_es = np.zeros((nt, npatch))
        self._p_cont_spr = np.zeros((nt, npatch))
        self._p_cont_dmp = np.zeros((nt, npatch))
        self._p_tot = np.zeros((nt, npatch))


class FixedStepSolver:
    
    Properties = namedlist('Properties', 'gap gap_eff')

    def __init__(self, t_fir, t_v, gap, gap_eff, t_lim, k, n, x0, lmbd, atol=1e-10, maxiter=5):

        fir_t, fir = t_fir
        t_start, t_stop, t_step = t_lim
        npatch = fir.shape[0]

        # define fir (with interpolation)
        fi_fir = interp1d(fir_t, fir, axis=-1, kind='cubic', assume_sorted=True)
        self._fir_t = np.arange(fir_t[0], fir_t[-1], t_step)
        self._fir = fi_fir(self._fir_t)

        # define patch properties
        self._gap = np.array(gap)
        self._gap_eff = np.array(gap_eff)

        # define patch state
        self._db = StateDB(t_lim, t_v, npatch)

        # create other variables
        self._error = []
        self._iters = []
        self.atol = atol
        self.maxiter = maxiter
        self.npatch = npatch
        self.current_step = 1
        self.min_step = t_step

        self._p_cont_spr = make_p_cont_spr(k, n, x0)
        self._p_cont_dmp = make_p_cont_dmp(lmbd, n, x0)

        # set initial state
        self._update_all(self.get_previous_state(), self.get_previous_state())

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
        return self._db.t[:self.current_step]

    @property
    def voltage(self):
        return self._db.v[:self.current_step, :]
    
    @property
    def displacement(self):
        return self._db.x[:self.current_step, :]
    
    @property
    def velocity(self):
        return self._db.u[:self.current_step, :]

    @property
    def pressure_electrostatic(self):
        return self._db.p_es[:self.current_step, :]

    @property
    def pressure_contact_spring(self):
        return self._db.p_cont_spr[:self.current_step, :]

    @property
    def pressure_contact_damper(self):
        return self._db.p_cont_dmp[:self.current_step, :]

    @property
    def pressure_total(self):
        return self._db.p_tot[:self.current_step, :]

    @property
    def error(self):
        return np.array(self._error[:self.current_step])

    @property
    def iters(self):
        return np.array(self._iters[:self.current_step])
    
    @property
    def props(self):
        return self.Properties(gap=self._gap, gap_eff=self._gap_eff)

    def get_current_state(self):
        return self._db.get_state_i(self.current_step)
    
    def get_previous_state(self):
        return self._db.get_state_i(self.current_step - 1)

    def get_state_i(self, i):
        return self._db.get_state_i(i)

    def get_state_d(self, d):
        return self._db.get_state_i(self.current_step + d)

    def get_state_t(self, t):
        return self._db.get_state_t(t)

    def _fir_conv(self, p, offset):
        return fir_conv_cy(self._fir, p, self.min_step, offset=offset)

    def _fir_conv_base(self, p):
        return fir_conv_cy(self._fir, p[:-1, :], self.min_step, offset=1)

    def _fir_conv_add(self, p):
        return fir_conv_cy(self._fir, p[-1:, :], self.min_step, offset=0)

    def _update_all(self, state, state_prev):
        
        db = self._db
        props = self.props

        u = (state.x - state_prev.x) / self.min_step
        p_es = p_es_pp(state.v, state.x, props.gap_eff)
        p_cont_spr = self._p_cont_spr(state.x)
        p_tot = p_cont_spr + state.p_cont_dmp + p_es
        db.set_state_i(db.State(i=state.i, u=u, p_es=p_es, p_cont_spr=p_cont_spr, 
                                p_tot=p_tot))

    def _update_cont_dmp(self, state):
        
        db = self._db

        p_cont_dmp = self._p_cont_dmp(state.x, state.u)
        db.set_state_i(db.State(i=state.i, p_cont_dmp=p_cont_dmp))

    def _update_x(self, conv_base=None):

        db = self._db
        state = self.get_current_state()

        p = self._db.p_tot[:self.current_step + 1, :]

        if conv_base is None:
            conv_base = self._fir_conv_base(p)

        conv_add = self._fir_conv_add(p)
        xnew = conv_base + conv_add
        err = np.max(np.abs(state.x - xnew))
        db.set_state_i(db.State(i=state.i, x=xnew))

        return  err, conv_base

    def _blind_x(self):

        db = self._db
        state = self.get_current_state()
        state_prev = self.get_previous_state()

        x = self._fir_conv(self._db.p_tot[:self.current_step, :], offset=1)
        db.set_state_i(db.State(i=state.i, x=x))
        self._update_all(state, state_prev)
        
    def step(self):

        db = self._db
        state = self.get_current_state()
        state_prev = self.get_previous_state()
        props = self.props

        # make blind estimate of displacement
        self._blind_x()

        # update displacement estimate without considering contact damping
        err, base = self._update_x(conv_base=None)
        self._update_all(state, state_prev)

        i = 2  # track number of convolutions done for calculation
        for j in range(self.maxiter - 1):
            if err <= self.atol:
                break

            err, base = self._update_x(conv_base=base)
            self._update_all(state, state_prev)
            i += 1

        # update contact damping based on current estimates
        self._update_cont_dmp(state)
        self._update_all(state, state_prev)
        
        # update displacement estimate with constant contact damping
        err, base = self._update_x(conv_base=base)
        self._update_all(state, state_prev)

        for j in range(self.maxiter - 1):
            if err <= self.atol:
                break

            err, base = self._update_x(conv_base=base)
            self._update_all(state, state_prev)
            i += 1
        
        # save results
        self._error.append(err)
        self._iters.append(i)
        self.current_step += 1
    
    def solve(self):

        stop = len(self._time) - 1
        while True:
            self.step()

            if self.current_step >= stop:
                break

    def reset(self):

        # reset state
        self._db.clear()
        self._error = []
        self._iters = []

        # set initial state
        self._update_all(self.get_previous_state(), self.get_previous_state())
        self.current_step = 1

    def __iter__(self):
        return self

    def __next__(self):

        if self.current_step >= len(self._time) - 1:
            raise StopIteration
        
        self.step()

        return self.current_step


class CompensationSolver(FixedStepSolver):

    def __init__(self, t_fir, t_v, gap, gap_eff, t_lim, comp_funcs, atol=1e-10, maxiter=5):

        self._comp_funcs = comp_funcs
        super().__init__(t_fir, t_v, gap, gap_eff, t_lim, 0, 0, 0, 0, atol=atol, maxiter=maxiter)

    @classmethod
    def from_array_and_db(cls, array, refn, dbfile, t_v, t_lim, lmbd, k, n=1, atol=1e-10, maxiter=5, **kwargs):

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

        comp_funcs = compensation.array_patch_comp_funcs(array, refn, lmbd, k, n, **kwargs)

        return cls((fir_t, fir), t_v, gap, gap_eff, t_lim, comp_funcs, atol, maxiter)

    def _update_all(self, state, state_prev):
        
        db = self._db
        comp_funcs = self._comp_funcs
        npatch = self.npatch

        u = (state.x - state_prev.x) / self.min_step

        p_es = np.zeros(npatch)
        p_cont_spr = np.zeros(npatch)

        for i in range(npatch):
            p_es[i] = comp_funcs[i]['p_es'](state.x[i], state.v[i])
            p_cont_spr[i] = comp_funcs[i]['p_cont_spr'](state.x[i])

        p_tot = p_cont_spr + state.p_cont_dmp + p_es

        db.set_state_i(db.State(i=state.i, u=u, p_es=p_es, p_cont_spr=p_cont_spr, 
                                p_tot=p_tot))

    def _update_cont_dmp(self, state):
        
        db = self._db
        comp_funcs = self._comp_funcs
        npatch = self.npatch

        p_cont_dmp = np.zeros(npatch)

        for i in range(npatch):
            p_cont_dmp[i] = comp_funcs[i]['p_cont_dmp'](state.x[i], state.u[i])

        db.set_state_i(db.State(i=state.i, p_cont_dmp=p_cont_dmp))

    @property
    def displacement_max(self):

        comp_funcs = self._comp_funcs
        x = self._db.x[:self.current_step, :]
        nt, npatch = x.shape
        xmax = np.zeros_like(x)

        for i in range(npatch):
            xmax[:, i] = comp_funcs[i]['xmax'](x[:, i])

        return xmax


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





        
    

    
