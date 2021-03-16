''''''
import numpy as np
from cnld import database, impulse_response, simulation

# def __init__(self, t_fir, t_v, gap, gap_eff, t_lim, k, n, x0, lmbd, atol=1e-10,
#                 maxiter=5):


class TimeSolver(simulation.FixedStepSolver):

    def __init__(self,
                 layout,
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
        '''
        Initialize solver from array object and its corresponding database.
        '''
        if calc_fir:
            # postprocess and convert frequency response to impulse response
            freqs, ppfr = database.read_patch_to_patch_freq_resp(dbfile)
            fir_t, fir = impulse_response.fft_to_fir(freqs,
                                                     ppfr,
                                                     interp=interp,
                                                     axis=-1,
                                                     use_kkr=use_kkr)

        else:
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

        super().__init__((fir_t, fir), t_v, gap, gap_eff, t_lim, k, n, x0, lmbd,
                         atol, maxiter)
