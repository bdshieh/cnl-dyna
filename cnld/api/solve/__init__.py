''''''
import numpy as np


class DatabaseSolver():
    def __init__(self, layout, geometry):
        pass


class FixedStepSolver():
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
        pass

    @classmethod
    def from_array_and_db(cls,
                          array,
                          dbfile,
                          t_v,
                          t_lim,
                          k,
                          n,
                          x0,
                          lmbd,
                          atol=1e-10,
                          maxiter=5,
                          calc_fir=False,
                          use_kkr=True,
                          interp=4):
        pass

    def _init_(self, layout, dbfile, transmit, k, n, x0, lmbd, atol, maxiter,
               calc_fir, use_kkr, interp):
        pass
