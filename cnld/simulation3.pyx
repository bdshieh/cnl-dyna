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

from cnld import abstract, impulse_response, compensation, database, fem

from simple_pid import PID


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


class FixedStepSolver:
    
    State = namedlist('State', 
    '''
    displacement velocity voltage pressure_applied pressure_electrostatic pressure_contact is_collapsed collapse_counter
    ''')
    Properties = namedlist('Properties', 'gap gap_effective, contact_stiffness, contact_damping')

    def __init__(self, t_fir, t_v, gap, gap_eff, t_lim, atol=1e-10, maxiter=5):

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
        fi_voltage = interp1d(v_t, v, axis=0, fill_value=0, bounds_error=False, kind='linear', assume_sorted=True)
        voltage = fi_voltage(time)

        # define fir (with interpolation)
        fi_fir = interp1d(fir_t, fir, axis=-1, kind='cubic', assume_sorted=True)
        self._fir_t = np.arange(fir_t[0], fir_t[-1], t_step)
        self._fir = fi_fir(self._fir_t)

        # define patch properties
        self._gap = np.array(gap)
        self._gap_eff = np.array(gap_eff)
        vmax = voltage.max(axis=0)
        self._cont_stiff = np.abs(electrostat_pres(vmax, -self._gap, self._gap_eff)) / 1e-9
        self._cont_damp = 0.05 * self._cont_stiff

        # define patch state
        self._time = time
        self._voltage = voltage
        self._displacement = np.zeros((ntime, npatch))
        self._velocity = np.zeros((ntime, npatch))
        self._pressure_electrostatic = np.zeros((ntime, npatch))
        self._pressure_contact = np.zeros((ntime, npatch))
        self._pressure_applied = np.zeros((ntime, npatch))
        self._is_collapsed = np.zeros((ntime, npatch), dtype=np.bool)
        self._collapse_counter = np.zeros((ntime, npatch))

        # create other variables
        self._error = []
        self._iters = []
        self.atol = atol
        self.maxiter = maxiter
        self.npatch = npatch
        self.ntime = ntime
        self.current_step = 0
        self.min_step = t_step

        self._pid = []
        for i in range(npatch):
            pid = PID(1.0, 0.2, 0.04)
            pid.setpoint = -self._gap[i]
            pid.auto_mode = False
            self._pid.append(pid)
        self._p0 = 0

        # self._pid2 = []
        # for i in range(npatch):
        #     pid = PID(1.0, 0.2, 0.04)
        #     pid.setpoint = -self._gap[i]
        #     pid.auto_mode = False
        #     self._pid2.append(pid)

        # set initial state
        self._update_pressure_applied(self.state_last, self.properties) 

    @classmethod
    def from_array_and_db(cls, array, dbfile, t_v, t_lim, atol=1e-10, maxiter=5):
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
                
        return cls((fir_t, fir), t_v, gap, gap_eff, t_lim, atol, maxiter)

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
        return np.array(self._displacement[:self.current_step + 1,:])

    @property
    def pressure_electrostatic(self):
        return np.array(self._pressure_electrostatic[:self.current_step + 1,:])

    @property
    def pressure_contact(self):
        return np.array(self._pressure_contact[:self.current_step + 1,:])

    @property
    def pressure_applied(self):
        return np.array(self._pressure_applied[:self.current_step + 1,:])
    
    @property
    def is_collapsed(self):
        return np.array(self._is_collapsed[:self.current_step + 1,:])

    @property
    def collapse_counter(self):
        return np.array(self._collapse_counter[:self.current_step + 1,:])

    @property
    def state(self):
        displacement = self.displacement
        velocity = self.velocity
        voltage = self.voltage
        pressure_applied = self.pressure_applied
        pressure_electrostatic = self.pressure_electrostatic
        pressure_contact = self.pressure_contact
        is_collapsed = self.is_collapsed
        collapse_counter = self.collapse_counter
        
        return self.State(displacement=displacement, velocity=velocity, voltage=voltage,
            pressure_applied=pressure_applied, pressure_electrostatic=pressure_electrostatic,
            pressure_contact=pressure_contact, is_collapsed=is_collapsed, 
            collapse_counter=collapse_counter)

    @property
    def state_last(self):
        idx = self.current_step
        displacement = self._displacement[idx,:]
        velocity = self._velocity[idx,:]
        voltage = self._voltage[idx,:]
        pressure_applied = self._pressure_applied[idx,:]
        pressure_electrostatic = self._pressure_electrostatic[idx,:]
        pressure_contact = self._pressure_contact[idx,:]
        is_collapsed = self._is_collapsed[idx,:]
        collapse_counter = self._collapse_counter[idx,:]

        return self.State(displacement=displacement, velocity=velocity, voltage=voltage,
            pressure_applied=pressure_applied, pressure_electrostatic=pressure_electrostatic,
            pressure_contact=pressure_contact, is_collapsed=is_collapsed, 
            collapse_counter=collapse_counter)
    
    @property
    def state_next(self):
        idx = self.current_step + 1
        displacement = self._displacement[idx,:]
        velocity = self._velocity[idx,:]
        voltage = self._voltage[idx,:]
        pressure_applied = self._pressure_applied[idx,:]
        pressure_electrostatic = self._pressure_electrostatic[idx,:]
        pressure_contact = self._pressure_contact[idx,:]
        is_collapsed = self._is_collapsed[idx,:]
        collapse_counter = self._collapse_counter[idx,:]

        return self.State(displacement=displacement, velocity=velocity, voltage=voltage,
            pressure_applied=pressure_applied, pressure_electrostatic=pressure_electrostatic,
            pressure_contact=pressure_contact, is_collapsed=is_collapsed, 
            collapse_counter=collapse_counter)

    @property
    def properties(self):
        return self.Properties(gap=self._gap, gap_effective=self._gap_eff, contact_stiffness=self._cont_stiff,
            contact_damping=self._cont_damp)

    def _fir_conv(self, p, offset):
        return fir_conv_cy(self._fir, p, self.min_step, offset=offset)

    def _update_pressure_electrostatic(self, state, props):

        pes = electrostat_pres(state.voltage, state.displacement, props.gap_effective)
        state.pressure_electrostatic[:] = pes

    def _update_pressure_contact(self, state, props):
        for i in range(self.npatch):
            if state.is_collapsed[i]:
                self._pid[i].auto_mode = True
                # pc1 = self._pid[i](state.displacement[i])
                
                # self._pid2[i].auto_mode = True
                # mask = self.is_collapsed[:,i]
                # if np.all(~mask):
                #     xmean = 0
                # else:
                #     x = self.displacement[:,i][mask]
                #     xmean = np.mean(x)
                # pc2 = self._pid2[i](xmean)

                # state.pressure_contact[i] = pc1 + pc2
                state.pressure_contact[i] = self._pid[i](state.displacement[i]) + self._p0

    def _update_pressure_applied(self, state, props):
        mask = state.is_collapsed
        state.pressure_applied[mask] = state.pressure_contact[mask]
        state.pressure_applied[~mask] = state.pressure_electrostatic[~mask]

        if self._time[self.current_step + 1] > 1e-6:
            state.pressure_applied[:] = state.pressure_electrostatic
        # state.pressure_applied[:] = state.pressure_contact + state.pressure_electrostatic

    def _update_velocity(self, state_last, state_next):
        state_next.velocity[:] = (state_next.displacement - state_last.displacement) / self.min_step

    def _check_for_collapse(self, state_last, state_next, props):

        cond1 = state_last.is_collapsed == True
        cond2 = state_next.displacement <= -props.gap + 10e-9
        mask = np.logical_or(cond1, cond2)
        state_next.is_collapsed[mask] = True

    def _blind_step(self):

        state = self.state
        state_last = self.state_last
        state_next = self.state_next
        props = self.properties

        state_next.displacement[:] = self._fir_conv(state.pressure_applied, offset=1)
        self._check_for_collapse(state_last, state_next, props)
        self._update_velocity(state_last, state_next)
        self._update_pressure_electrostatic(state_next, props)
        self._update_pressure_contact(state_next, props)
        self._update_pressure_applied(state_next, props)
 
    def _check_accuracy_of_step(self):
        
        state_next = self.state_next

        p = self._pressure_applied[:self.current_step + 2,:]

        xr = self._fir_conv(p, offset=0)
        err = np.max(np.abs(state_next.displacement - xr))

        return xr, err
        
    def step(self):

        state = self.state
        state_last = self.state_last
        state_next = self.state_next
        props = self.properties

        self._blind_step()
        xr, err = self._check_accuracy_of_step()

        i = 1
        for j in range(self.maxiter - 1):
            if err <= self.atol:
                break

            state_next.displacement[:] = xr
            self._check_for_collapse(state_last, state_next, props)
            self._update_velocity(state_last, state_next)
            self._update_pressure_electrostatic(state_next, props)
            self._update_pressure_contact(state_next, props)
            self._update_pressure_applied(state_next, props)
            
            xr, err = self._check_accuracy_of_step()
            i += 1

        self._error.append(err)
        self._iters.append(i)

        state_next.displacement[:] = xr
        self._check_for_collapse(state_last, state_next, props)
        self._update_velocity(state_last, state_next)
        self._update_pressure_electrostatic(state_next, props)
        self._update_pressure_contact(state_next, props)
        self._update_pressure_applied(state_next, props)
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
        self._pressure_electrostatic = np.zeros((ntime, npatch))
        self._pressure_contact = np.zeros((ntime, npatch))
        self._pressure_applied = np.zeros((ntime, npatch))

        # set initial state
        self._update_pressure_applied(self.state_last, self.properties) 
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





        
    

    
