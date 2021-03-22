'''Routines for time-domain simulation.'''
import numpy as np
# import scipy as sp
# import scipy.signal
from scipy.constants import epsilon_0 as e_0
from scipy.interpolate import interp1d
import numba
from namedlist import namedlist
import warnings

warnings.filterwarnings("ignore")


@numba.vectorize(cache=True, nopython=True)
def p_es_pp(v, x, g_eff):
    return -e_0 / 2 * v**2 / (x + g_eff)**2


@numba.njit(cache=True)
def fir_conv_cy(fir, p, dt, offset):

    nsrc, ndest, nfir = fir.shape
    nsample = p.shape[0]
    x = np.zeros(ndest, dtype=np.float64)
    ndot = min(nsample, nfir - offset)
    prev = np.flipud(p)

    for j in range(ndest):

        c = 0
        for i in range(nsrc):

            s = fir[i, j, offset:(ndot + offset)] @ prev[:ndot, i]
            c += s

        x[j] = c

    return x * dt


# @numba.njit(cache=True)
# def fir_conv2(fir, p, dt, offset):
#     '''Matrix math version, not faster ...'''
#     nsrc, ndest, nfir = fir.shape
#     nsample, _ = p.shape
#     ndot = min(nsample, nfir - offset)
#     prev = p[::-1, :].T

#     x = np.sum(np.sum(fir[:, :, offset:(ndot - offset)] * prev[:, :ndot], axis=2), axis=0)

#     return x * dt


def make_p_cont_spr(k, n, x0):

    @numba.vectorize(cache=True, nopython=True)
    def _p_cont_spr(x):
        if x >= x0:
            return 0
        return k * (x0 - x)**n

    return _p_cont_spr


def make_p_cont_dmp(lmbd, n, x0):
    '''
    Generate contact damper function.
    '''

    @numba.vectorize(cache=True, nopython=True)
    def _p_cont_dmp(x, xdot):
        if x >= x0:
            return 0
        return -(lmbd * (x0 - x)**n) * xdot

    return _p_cont_dmp


class StateDB:
    '''
    Simple database to store state variables.
    '''
    State = namedlist('State',
                      'i t x u v p_tot p_es p_cont_spr p_cont_dmp',
                      default=None)

    def __init__(self, t_lim, t_v, npatch):

        v_t, v = t_v
        t_start, t_stop, t_step = t_lim

        # define time array
        t = np.arange(t_start, t_stop + t_step, t_step)
        nt = len(t)

        # define voltage array (with interpolation)
        if v.ndim <= 1:
            v = np.tile(v, (npatch, 1)).T
        fi_voltage = interp1d(v_t,
                              v,
                              axis=0,
                              fill_value=(v[0, :], v[-1, :]),
                              bounds_error=False,
                              kind='linear',
                              assume_sorted=True)
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
        '''
        Sample index.
        '''
        arr = self._i.view()
        # arr.flags.writeable = False
        return arr

    @property
    def t(self):
        '''
        Time.
        '''
        arr = self._t.view()
        # arr.flags.writeable = False
        return arr

    @property
    def v(self):
        '''
        Voltage.
        '''
        arr = self._v.view()
        # arr.flags.writeable = False
        return arr

    @property
    def x(self):
        '''
        Displacement.
        '''
        arr = self._x.view()
        # arr.flags.writeable = False
        return arr

    @property
    def u(self):
        '''
        Velocity.
        '''
        arr = self._u.view()
        # arr.flags.writeable = False
        return arr

    @property
    def p_es(self):
        '''
        Electrostatic pressure.
        '''
        arr = self._p_es.view()
        # arr.flags.writeable = False
        return arr

    @property
    def p_cont_spr(self):
        '''
        Contact spring pressure.
        '''
        arr = self._p_cont_spr.view()
        # arr.flags.writeable = False
        return arr

    @property
    def p_cont_dmp(self):
        '''
        Contact damper pressure.
        '''
        arr = self._p_cont_dmp.view()
        # arr.flags.writeable = False
        return arr

    @property
    def p_tot(self):
        '''
        Total pressure.
        '''
        arr = self._p_tot.view()
        # arr.flags.writeable = False
        return arr

    def get_state_i(self, i):
        '''
        Return state at index i.
        '''
        return self.State(i=self.i[i],
                          t=self.t[i],
                          v=self.v[i, :],
                          x=self.x[i, :],
                          u=self.u[i, :],
                          p_es=self.p_es[i, :],
                          p_cont_spr=self.p_cont_spr[i, :],
                          p_cont_dmp=self.p_cont_dmp[i, :],
                          p_tot=self.p_tot[i, :])

    def get_state_t(self, t):
        '''
        Return state at time t.
        '''
        i = int(np.argmin(np.abs(t - self.t)))
        return self.get_state_i(i)

    def set_state_i(self, s):
        '''
        Set state at index i.
        '''
        i = s.i

        if s.x is not None:
            self._x[i, :] = s.x
        if s.u is not None:
            self._u[i, :] = s.u
        if s.p_es is not None:
            self._p_es[i, :] = s.p_es
        if s.p_cont_spr is not None:
            self._p_cont_spr[i, :] = s.p_cont_spr
        if s.p_cont_dmp is not None:
            self._p_cont_dmp[i, :] = s.p_cont_dmp
        if s.p_tot is not None:
            self._p_tot[i, :] = s.p_tot

    def set_state_t(self, s):
        '''
        Set state at time t.
        '''
        t = s.t
        i = int(np.min(np.abs(t - self._t)))
        self.set_state_i(i)

    def clear(self):
        '''
        Clear database.
        '''
        nt, npatch = self._x.shape

        self._x = np.zeros((nt, npatch))
        self._u = np.zeros((nt, npatch))
        self._p_es = np.zeros((nt, npatch))
        self._p_cont_spr = np.zeros((nt, npatch))
        self._p_cont_dmp = np.zeros((nt, npatch))
        self._p_tot = np.zeros((nt, npatch))


class FixedStepSolver:

    Properties = namedlist('Properties', 'gap gap_eff')

    def __init__(self,
                 t_fir,
                 t_v,
                 gap,
                 gap_eff,
                 t_lim,
                 k,
                 n,
                 x0,
                 lmbd,
                 atol=1e-10,
                 maxiter=5):

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
        # convolves input with LTI system
        return fir_conv_cy(self._fir, p, self.min_step, offset=offset)

    def _fir_conv_base(self, p):
        # convolves input with LTI system for all previous time steps
        return fir_conv_cy(self._fir, p[:-1, :], self.min_step, offset=1)

    def _fir_conv_add(self, p):
        # convolves input with LTI system for only the current time step
        return fir_conv_cy(self._fir, p[-1:, :], self.min_step, offset=0)

    def _update_all(self, state, state_prev):
        # update all state variables for the given state
        db = self._db
        props = self.props

        u = (state.x - state_prev.x) / self.min_step
        p_es = p_es_pp(state.v, state.x, props.gap_eff)
        p_cont_spr = self._p_cont_spr(state.x)
        p_tot = p_cont_spr + state.p_cont_dmp + p_es
        db.set_state_i(
            db.State(i=state.i,
                     u=u,
                     p_es=p_es,
                     p_cont_spr=p_cont_spr,
                     p_tot=p_tot))

    def _update_cont_dmp(self, state):
        # update contact damper pressure for the given state
        db = self._db

        p_cont_dmp = self._p_cont_dmp(state.x, state.u)
        db.set_state_i(db.State(i=state.i, p_cont_dmp=p_cont_dmp))

    def _update_x(self, conv_base=None):
        # update displacement via fixed-point iteration
        db = self._db
        state = self.get_current_state()

        p = self._db.p_tot[:self.current_step + 1, :]

        # conv_base is calculated only once to reduce computations drastically
        if conv_base is None:
            conv_base = self._fir_conv_base(p)

        conv_add = self._fir_conv_add(p)
        xnew = conv_base + conv_add
        err = np.max(np.abs(state.x - xnew))
        db.set_state_i(db.State(i=state.i, x=xnew))

        return err, conv_base

    def _blind_x(self):
        # blind estiamte of displacement
        db = self._db
        state = self.get_current_state()
        state_prev = self.get_previous_state()

        x = self._fir_conv(self._db.p_tot[:self.current_step, :], offset=1)
        db.set_state_i(db.State(i=state.i, x=x))
        self._update_all(state, state_prev)

    def step(self):
        '''
        Step solver in time by one sample.
        '''
        # db = self._db
        state = self.get_current_state()
        state_prev = self.get_previous_state()
        # props = self.props

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
        '''
        Solve for all time samples.
        '''
        stop = len(self._db.t)
        while True:
            self.step()

            if self.current_step >= stop:
                break

    def reset(self):
        '''
        Reset solver to initial state.
        '''
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

        if self.current_step >= len(self._db.t):
            raise StopIteration

        self.step()

        return self.current_step

    def __len__(self):
        return len(self._db.t) - 2


# class CompensationSolver(FixedStepSolver):
#     '''
#     *Experimental* Time-domain solver with deflection profile compensation.

#     Parameters
#     ----------
#     FixedStepSolver : [type]
#         [description]
#     '''

#     def __init__(self,
#                  t_fir,
#                  t_v,
#                  gap,
#                  gap_eff,
#                  t_lim,
#                  comp_funcs,
#                  atol=1e-10,
#                  maxiter=5):
#         '''
#         [summary]

#         Parameters
#         ----------
#         t_fir : [type]
#             [description]
#         t_v : [type]
#             [description]
#         gap : [type]
#             [description]
#         gap_eff : [type]
#             [description]
#         t_lim : [type]
#             [description]
#         comp_funcs : [type]
#             [description]
#         atol : [type], optional
#             [description], by default 1e-10
#         maxiter : int, optional
#             [description], by default 5
#         '''
#         self._comp_funcs = comp_funcs
#         super().__init__(t_fir,
#                          t_v,
#                          gap,
#                          gap_eff,
#                          t_lim,
#                          0,
#                          0,
#                          0,
#                          0,
#                          atol=atol,
#                          maxiter=maxiter)

#     @classmethod
#     def from_array_and_db(cls,
#                           array,
#                           refn,
#                           dbfile,
#                           t_v,
#                           t_lim,
#                           lmbd,
#                           k,
#                           n=1,
#                           atol=1e-10,
#                           maxiter=5,
#                           **kwargs):

#         # read fir database
#         fir_t, fir = database.read_patch_to_patch_imp_resp(dbfile)

#         # create gap and gap eff
#         gap = []
#         gap_eff = []
#         for elem in array.elements:
#             for mem in elem.membranes:
#                 for pat in mem.patches:
#                     gap.append(mem.gap)
#                     gap_eff.append(mem.gap + mem.isolation / mem.permittivity)

#         comp_funcs = compensation.array_patch_comp_funcs(
#             array, refn, lmbd, k, n, **kwargs)

#         return cls((fir_t, fir), t_v, gap, gap_eff, t_lim, comp_funcs, atol,
#                    maxiter)

#     def _update_all(self, state, state_prev):

#         db = self._db
#         comp_funcs = self._comp_funcs
#         npatch = self.npatch

#         u = (state.x - state_prev.x) / self.min_step

#         p_es = np.zeros(npatch)
#         p_cont_spr = np.zeros(npatch)

#         for i in range(npatch):
#             p_es[i] = comp_funcs[i]['p_es'](state.x[i], state.v[i])
#             p_cont_spr[i] = comp_funcs[i]['p_cont_spr'](state.x[i])

#         p_tot = p_cont_spr + state.p_cont_dmp + p_es

#         db.set_state_i(
#             db.State(i=state.i,
#                      u=u,
#                      p_es=p_es,
#                      p_cont_spr=p_cont_spr,
#                      p_tot=p_tot))

#     def _update_cont_dmp(self, state):

#         db = self._db
#         comp_funcs = self._comp_funcs
#         npatch = self.npatch

#         p_cont_dmp = np.zeros(npatch)

#         for i in range(npatch):
#             p_cont_dmp[i] = comp_funcs[i]['p_cont_dmp'](state.x[i], state.u[i])

#         db.set_state_i(db.State(i=state.i, p_cont_dmp=p_cont_dmp))

#     @property
#     def displacement_max(self):

#         comp_funcs = self._comp_funcs
#         x = self._db.x[:self.current_step, :]
#         nt, npatch = x.shape
#         xmax = np.zeros_like(x)

#         for i in range(npatch):
#             xmax[:, i] = comp_funcs[i]['xmax'](x[:, i])

#         return xmax