''''''
__all__ = ['PressurePointSolver']
import numpy as np
import scipy as sp
from scipy import signal
from cnld import bem


class PressurePointSolver:

    def __init__(self, sir, dt):
        self._sir = sir
        self._dt = dt

    @classmethod
    def from_layout(cls,
                    layout,
                    grid,
                    db_file,
                    r,
                    c,
                    rho,
                    use_kkr=False,
                    interp=2):

        sir_t, sir = bem.pfir_from_layout(layout,
                                          grid,
                                          db_file,
                                          r,
                                          c,
                                          rho,
                                          interp=interp,
                                          use_kkr=use_kkr)
        sir = sir.T
        dt = sir_t[1] - sir_t[0]
        return cls(sir, dt)

    @classmethod
    def from_npz(cls, file):
        with np.load(file) as npz:
            sir = npz['sir']
            sir_t = npz['sir_t']

        dt = sir_t[1] - sir_t[0]
        return cls(sir, dt)

    def solve(self, pes):

        sir = self._sir
        dt = self._dt
        return np.sum(sp.signal.fftconvolve(pes, sir, axis=0) * dt, axis=1)

    def export_npz(self, file):

        sir = self._sir
        dt = self._dt
        np.savez(file, sir=sir, dt=dt)