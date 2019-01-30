

import numpy as np
import scipy as sp
import scipy.signal
from scipy.fftpack import fft, ifft, fftshift, ifftshift, fftfreq
from matplotlib import pyplot as plt
from scipy.constants import epsilon_0 as e_0
from tqdm import tqdm

# from cnld import simulation, impulse_response
from cnld import impulse_response


def pressure_es(v, x, g_eff):
    return -e_0 / 2 * v**2 / (x + g_eff)**2
    

def convolve_fir(A, b, fs):
    nsrc, ndest, nfir = A.shape
    _, nsample = b.shape
    # x = np.zeros(ndest)

    # if b.ndim == 1:
        # b = b[...,None]

    # n = min(nsample, nfir)
    # x = np.sum(np.sum(A[:,:,:n] * b[:,None,::-1], axis=-1), axis=0).squeeze()
    # for i in range(nsrc):
        # for j in range(ndest):
            # x[i,j] += np.sum(A[i,j,:nsample] * b[i,::-1]) / fs
    x = np.zeros(ndest)
    for j in range(ndest):
        c = 0
        for i in range(nsrc):
            ir = A[i,j,:]
            f = b[i,:]
            frev = f[::-1]

            s = 0
            for k in range(min(nsample, nfir - 1)):
                s += ir[k + 1] * frev[k]
            
            # s = ir[-1] * frev[0] / fs
            # s = np.convolve(ir, f)[-1]

            c += s

        x[j] = c / fs

    return x


def matrix_convolve(A, b, fs):
    nsrc, ndest, nfir = A.shape
    _, nsample = b.shape
    conv = []
    for i in range(ndest):
        _conv = 0
        for j in range(nsrc):
            _conv += np.convolve(A[j,i,:], b[j,:], mode='full')[:nsample] / fs
        conv.append(_conv)

    return np.array(conv)


def gausspulse(fc, fbw, fs, tpr=-100, retquad=False):
    cutoff = sp.signal.gausspulse('cutoff', fc=fc, bw=fbw, tpr=tpr, bwr=-3)
    adj_cutoff = np.ceil(cutoff * fs) / fs

    t = np.arange(-adj_cutoff, adj_cutoff + 1 / fs, 1 / fs)
    pulse, quad = sp.signal.gausspulse(t, fc=fc, bw=fbw, retquad=True, bwr=-3)

    return t, pulse


class Solver:
    
    def __init__(self, fir, v, x0, tstart, tstop, hmin):
        
        t = np.arange(tstart, tstop, hmin)
        npatch = fir.shape[0]
        x = np.zeros((npatch, len(t)))
        x[:,0] = x0
        f = np.zeros((npatch, len(t)))
        # f = 20 * np.ones((npatch, len(t)))
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

        fs = 1 / hmin

        g_eff = 50e-9
        p_es = pressure_es(v[ti], x[:,ti], g_eff)
        f[:,ti] = p_es

        x_new = convolve_fir(fir, f[:,:(ti + 1)], fs)
        # x_new = matrix_convolve(fir, f[:,:(ti + 1)], fs)[..., -1]
        x[:,ti + 1] = x_new

        self.ti += 1
        

def one_to_two(f, s, axis=-1):
    '''
    '''
    s = np.atleast_2d(s)

    nf = s.shape[axis]
    nfft = nf * 2 - 1

    newshape = list(s.shape)
    newshape[axis] = nfft

    s2s = np.zeros(newshape, dtype=np.complex128)

    idx1 = [slice(None)] * s.ndim
    idx1[axis] = slice(None, nfft // 2 + 1)
    idx2 = [slice(None)] * s.ndim
    idx2[axis] = slice(None, None, None)
    s2s[tuple(idx1)] = s[tuple(idx2)]
    # s2s[:, :nfft / 2] = s[:, :-1]

    idx1 = [slice(None)] * s.ndim
    idx1[axis] = slice(nfft // 2 + 1, None, None)
    idx2 = [slice(None)] * s.ndim
    idx2[axis] = slice(-1, 0, -1)
    s2s[tuple(idx1)] = np.conj(s[tuple(idx2)])
    # s2s[:, nfft / 2:] = np.conj(s[:, -1:0:-1])

    df = f[1] - f[0]
    fs = df * nfft
    f2s = fftfreq(nfft, 1 / fs)

    return f2s, s2s


freqs, ginv = impulse_response.read_freq_resp_db('freq_resp.db')
# freqs = np.insert(freqs, 0, 0)
# ginv = np.insert(ginv, 0, np.abs(ginv[:,:,1]), axis=-1)
# ginv[...,0] = 0
freqs2, ginv2 = one_to_two(freqs, ginv)
fs = (freqs2[1] - freqs2[0]) * len(freqs2)
fir = np.real(ifft(np.conj(ginv2), axis=-1)) * fs
# fir = ifftshift(fftshift(fir, axes=-1) * np.hanning(fir.shape[2]), axes=-1)
fir = fir * np.hanning(fir.shape[2] * 2 + 1)[fir.shape[2]:-1]
# fir = np.real(ifft(ginv, axis=-1)) * fs
npatch = fir.shape[0]

tstart = 0
tstop = 10e-6
hmin = 1 / fs
t = np.arange(tstart, tstop, hmin)
vdc = 10 / (1 + np.exp(-5e6 * (t - 1e-6)))
# vdc = 20 * np.ones(len(t))
# vac = -20 * gausspulse(4e6, 1.2, fs, tpr=-80, retquad=True)
# vac = np.roll(np.pad(vac, ((0, len(vdc) - len(vac))), mode='constant'), 300)
# v = vdc + vac
v = vdc
# v = 40 * np.sin(2 * np.pi * 1e6 * t)
# v = 40 * gausspulse(7e6, 1, fs=fs)[:len(t)]
# v = 20 * sp.signal.square(2 * np.pi * 1e6 * t)
# v = np.pad(v, (20, 0), mode='constant')[:-20]
x0 = np.zeros(npatch)

hmin = 1 / fs

solver = Solver(fir, v, x0, tstart, tstop, hmin)
# solver.f[:,:] = vdc

for i in tqdm(range(len(v) - 1)):
    solver.step()



fig, ax = plt.subplots()
ax.plot(t, solver.x[4,:] / 1e-9)
tax = ax.twinx()
tax.plot(t, v, '--', color='orange')
# plt.plot(t, solver.x[13,:])
fig.show()

fig, ax = plt.subplots()
ax.plot(t[:-1], solver.f[4,:-1])
tax = ax.twinx()
tax.plot(t, v, '--', color='orange')
# plt.plot(t, solver.x[13,:])
fig.show()