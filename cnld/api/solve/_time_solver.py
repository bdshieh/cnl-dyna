''''''
__all__ = ['TimeSolver']
import numpy as np
from itertools import cycle
from cnld import database, impulse_response, simulation

# def __init__(self, t_fir, t_v, gap, gap_eff, t_lim, k, n, x0, lmbd, atol=1e-10,
#                 maxiter=5):

# class TimeSolver(simulation.FixedStepSolver):

#     def __init__(self,
#                  layout,
#                  transmit,
#                  dbfile,
#                  times,
#                  atol=1e-10,
#                  maxiter=5,
#                  calc_fir=False,
#                  use_kkr=True,
#                  interp=4):
#         '''
#         Initialize solver from array object and its corresponding database.
#         '''
#         if calc_fir:
#             # postprocess and convert frequency response to impulse response
#             freqs, ppfr = database.read_patch_to_patch_freq_resp(dbfile)
#             fir_t, fir = impulse_response.fft_to_fir(freqs,
#                                                      ppfr,
#                                                      interp=interp,
#                                                      axis=-1,
#                                                      use_kkr=use_kkr)

#         else:
#             # read fir database
#             fir_t, fir = database.read_patch_to_patch_imp_resp(dbfile)

#         # create gap and gap eff
#         gap = [None] * len(layout.controldomains)
#         gap_eff = [None] * len(layout.controldomains)

#         if layout.membrane_to_geometry_mapping is None:
#             gid = cycle(range(len(layout.geometries)))
#             mapping = [next(gid) for i in range(len(layout.membranes))]

#         for i, ctrldom in enumerate(layout.controldomains):
#             geom = layout.geometries[mapping[ctrldom.membrane_id]]
#             gap[i] = geom.gap
#             gap_eff[i] = geom.gap + geom.isol_thickness / geom.eps_r

#         nelem = len(layout.elements)
#         if transmit.apod is None:
#             apod = np.ones(nelem)
#         if transmit.delays is None:
#             delays = np.zeros(nelem, dtype=int)

#         if transmit.element_to_waveform_mapping is None:
#             wid = cycle(range(len(transmit.waveforms)))
#             wf_mapping = [next(wid) for i in range(nelem)]

#         waveforms = [None] * nelem
#         for i in range(nelem):
#             wf = transmit.waveforms[wf_mapping[i]]
#             waveforms[i] = apod[i] + np.pad(wf.voltage, (delays[i], 0))

#         waveforms = concatenate_with_padding(*waveforms)
#         t = np.arange(waveforms.shape[0]) / transmit.fs

#         v = np.zeros((waveforms.shape[0], len(layout.controldomains)))
#         for i, ctrldom in enumerate(layout.controldomains):
#             v[:, i] = waveforms[:, ctrldom.element_id]
#         # for elem in layout.elements:
#         #     for mid in elem.membrane_ids:
#         #         idx = ctrldomlist.id[ctrldomlist.membrane_id == mid]
#         #         v[:, idx] = waveforms[:, elem.id]

#         # lazy support for one set of contact parameters
#         k = layout.geometries[0].contact_k
#         n = layout.geometries[0].contact_n
#         x0 = layout.geometries[0].contact_z0
#         lmbd = layout.geometries[0].contact_lmda

#         super().__init__((fir_t, fir), (t, v), gap, gap_eff, times, k, n, x0,
#                          lmbd, atol, maxiter)

#     @property
#     def layout(self):
#         return self._layout

#     @property
#     def transmit(self):
#         return self._transmit

#     @property
#     def times(self):
#         return self._times

#     @property
#     def dbfile(self):
#         return self._dbfile


class TimeSolver():

    def __init__(self,
                 layout=None,
                 transmit=None,
                 dbfile=None,
                 times=None,
                 atol=1e-10,
                 maxiter=5):

        self.layout = layout
        self.transmit = transmit
        self.times = times
        self.atol = atol
        self.maxiter = maxiter
        self.dbfile = dbfile

    @property
    def layout(self):
        return self._layout

    @layout.setter
    def layout(self, arg):
        self._layout = arg

    @property
    def transmit(self):
        return self._transmit

    @transmit.setter
    def transmit(self, arg):
        self._transmit = arg

    @property
    def times(self):
        return self._times

    @times.setter
    def times(self, arg):
        self._times = arg

    @property
    def dbfile(self):
        return self._dbfile

    @dbfile.setter
    def dbfile(self, f):
        if f is not None:
            fir_t, fir = database.read_patch_to_patch_imp_resp(f)
            self._fir = fir
            self._fir_t = fir_t
            self._dbfile = f

    @property
    def atol(self):
        return self._atol

    @atol.setter
    def atol(self, arg):
        self._atol = arg

    @property
    def maxiter(self):
        return self._maxiter

    @maxiter.setter
    def maxiter(self, arg):
        self._maxiter = arg

    def recalculate_fir(self, use_kkr=True, interp=4):

        dbfile = self.dbfile

        # postprocess and convert frequency response to impulse response
        freqs, ppfr = database.read_patch_to_patch_freq_resp(dbfile)
        fir_t, fir = impulse_response.fft_to_fir(freqs,
                                                 ppfr,
                                                 interp=interp,
                                                 axis=-1,
                                                 use_kkr=use_kkr)

        # remove uneccessary second half due to kkr
        if use_kkr:
            nfir = len(fir_t)
            fir_t = fir_t[:(nfir // 2)]
            fir = fir[..., :(nfir // 2)]

        self._fir = fir
        self._fir_t = fir_t

    def setup(self):
        '''
        Initialize solver from array object and its corresponding database.
        '''
        layout = self.layout
        transmit = self.transmit
        times = self.times
        atol = self.atol
        maxiter = self.maxiter
        fir = self._fir
        fir_t = self._fir_t

        # create gap and gap eff
        gap = [None] * len(layout.controldomains)
        gap_eff = [None] * len(layout.controldomains)

        if layout.membrane_to_geometry_mapping is None:
            gid = cycle(range(len(layout.geometries)))
            mapping = [next(gid) for i in range(len(layout.membranes))]

        for i, ctrldom in enumerate(layout.controldomains):
            geom = layout.geometries[mapping[ctrldom.membrane_id]]
            gap[i] = geom.gap
            gap_eff[i] = geom.gap + geom.isol_thickness / geom.eps_r

        nelem = len(layout.elements)
        if transmit.apod is None:
            apod = np.ones(nelem)
        if transmit.delays is None:
            delays = np.zeros(nelem, dtype=int)

        if transmit.element_to_waveform_mapping is None:
            wid = cycle(range(len(transmit.waveforms)))
            wf_mapping = [next(wid) for i in range(nelem)]

        waveforms = [None] * nelem
        for i in range(nelem):
            wf = transmit.waveforms[wf_mapping[i]]
            waveforms[i] = apod[i] + np.pad(wf.voltage, (delays[i], 0))

        waveforms = concatenate_with_padding(*waveforms)
        t = np.arange(waveforms.shape[0]) / transmit.fs

        v = np.zeros((waveforms.shape[0], len(layout.controldomains)))
        for i, ctrldom in enumerate(layout.controldomains):
            v[:, i] = waveforms[:, ctrldom.element_id]

        # lazy support for one set of contact parameters
        k = layout.geometries[0].contact_k
        n = layout.geometries[0].contact_n
        x0 = layout.geometries[0].contact_z0
        lmbd = layout.geometries[0].contact_lmda

        self._solver = simulation.FixedStepSolver(
            (fir_t, fir), (t, v), gap, gap_eff, times, k, n, x0, lmbd, atol,
            maxiter)

    def step(self):
        self._solver.step()

    def solve(self):
        self._solver.solve()

    def reset(self):
        self._solver.reset()

    def __iter__(self):
        return self._solver.__iter__()

    def __len__(self):
        return len(self._solver)

    def __next__(self):
        return self._solver.__next__()

    @property
    def time(self):
        return self._solver.time

    @property
    def voltage(self):
        return self._solver.voltage

    @property
    def displacement(self):
        return self._solver.displacement

    @property
    def velocity(self):
        return self._solver.velocity

    @property
    def pressure_electrostatic(self):
        return self._solver.pressure_electrostatic

    @property
    def pressure_contact_spring(self):
        return self._solver.pressure_contact_spring

    @property
    def pressure_contact_damper(self):
        return self._solver.pressure_contact_damper

    @property
    def pressure_total(self):
        return self._solver.pressure_total

    @property
    def error(self):
        return self._solver.error

    @property
    def iters(self):
        return self._solver.iters


def concatenate_with_padding(*data):

    maxlen = max([len(d) for d in data])
    backpads = [maxlen - len(d) for d in data]

    new_data = [None] * len(data)

    for i, (d, bpad) in enumerate(zip(data, backpads)):

        new_data[i] = np.pad(d, ((0, bpad)), mode='constant')

    return np.stack(new_data, axis=1)
