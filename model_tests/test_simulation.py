

import numpy as np
import scipy as sp
from scipy.fftpack import fft, ifft
from matplotlib import pyplot as plt
from scipy.constants import epsilon_0 as e_0

# from cnld import simulation, impulse_response
from cnld import impulse_response


def pressure_es(v, x, g_eff):
    '''
    '''
    return -e_0 / 2 * v**2 / (x + g_eff)**2
    

def convolve_fir(A, b, fs):
    '''
    '''
    nsrc, ndest, nfir = A.shape
    _, nsample = b.shape
    # x = np.zeros(ndest)

    if b.ndim == 1:
        b = b[...,None]

    x = np.sum(np.sum(A[:,:,:nsample] * b[:,None,::-1], axis=-1), axis=0).squeeze()
    # for i in range(nsrc):
        # for j in range(ndest):
            # x[i,j] += np.sum(A[i,j,:nsample] * b[i,::-1]) / fs
    return x


def gausspulse(fc, fbw, fs, tpr=-100, retquad=False):
    '''
    '''
    cutoff = sp.signal.gausspulse('cutoff', fc=fc, bw=fbw, tpr=tpr, bwr=-3)
    adj_cutoff = np.ceil(cutoff * fs) / fs

    t = np.arange(-adj_cutoff, adj_cutoff + 1 / fs, 1 / fs)
    pulse, quad = sp.signal.gausspulse(t, fc=fc, bw=fbw, retquad=True, bwr=-3)
    
    if retquad:
        return quad
    
    return pulse


class Solver:
    
    def __init__(self, fir, v, x0, tstart, tstop, hmin):
        
        t = np.arange(tstart, tstop, hmin)
        npatch = fir.shape[0]
        x = np.zeros((npatch, len(t)))
        x[:,0] = x0
        f = np.zeros((npatch, len(t)))
        # hmax = round((tstop - tstart) / 50)

        self.t = t
        self.v = v
        self.x = x
        self.f = f
        self.fir = fir
        self.hmin = hmin
        self.ti = 0
        # self.hmax = hmax
    
    def step(self):
        
        ti = self.ti
        v = self.v
        x = self.x
        f = self.f
        fir = self.fir
        hmin = self.hmin

        g_eff = 100e-9
        p_es = pressure_es(v[ti], x[:,ti], g_eff)
        f[:,ti] = p_es

        x_new = convolve_fir(fir, f[:,:(ti + 1)], fs=(1 / hmin))
        x[:,ti + 1] = x_new

        self.ti += 1
        

freqs, ginv = impulse_response.read_freq_resp_db('freq_resp.db')
freqs2, ginv2 = impulse_response.one_to_two(freqs, ginv)
fs = (freqs2[1] - freqs2[0]) * len(freqs2)
fir = np.real(ifft(ginv, axis=-1)) * fs
npatch = fir.shape[0]

tstart = 0
tstop = 10e-6
hmin = 1 / fs
t = np.arange(tstart, tstop, hmin)
v = 20 * np.sin(2 * np.pi * 1e6 * t)
x0 = np.zeros(npatch)

hmin = 1 / fs

solver = Solver(fir, v, x0, tstart, tstop, hmin)

for i in range(100):
    solver.step()


plt.plot(t, solver.x[4,:])
plt.show()