

import numpy as np
import scipy as sp
import scipy.signal
from scipy.fftpack import fft, ifft, fftshift, ifftshift, fftfreq
from matplotlib import pyplot as plt
from scipy.constants import epsilon_0 as e_0
from tqdm import tqdm
from scipy.interpolate import interp1d

# from cnld import simulation, impulse_response
from cnld import impulse_response, frequency_response


def pressure_es(v, x, g_eff):
    return -e_0 / 2 * v**2 / (x + g_eff)**2
    

def firconvolve(fir, p, fs, offset):
    nsrc, ndest, nfir = fir.shape
    nsample, _ = p.shape

    x = np.zeros(ndest)
    for j in range(ndest):
        c = 0
        for i in range(nsrc):
            ir = A[i,j,:]
            f = b[:,i]
            frev = f[::-1]

            s = 0
            for k in range(min(nsample, nfir - offset)):
                s += ir[k + offset] * frev[k]
            c += s

        x[j] = c / fs
    return x






class Solver:
    
    def __init__(self, v, v_t, x0, g_eff, delay):
        
        freqs, ginv = frequency_response.read_db('freq_resp.db')
        # win = np.hanning(ginv.shape[2] * 2 + 1)[ginv.shape[2]:-1]
        # win = sp.signal.tukey(ginv.shape[2] * 2 + 1, alpha=0.1)[ginv.shape[2]:-1]
        # ginv = ginv * win
        freqs2, ginv2 = one_to_two(freqs, ginv)
        fs = (freqs2[1] - freqs2[0]) * len(freqs2)
        ginv2_causal = np.real(ginv2) - 1j * np.imag(sp.signal.hilbert(np.real(ginv2), axis=-1))
        fir = np.real(ifft(ginv2_causal, axis=-1)) * fs
        # fir = np.real(ifft(np.conj(ginv2), axis=-1)) * fs
        fir = np.roll(fir, delay, axis=-1)
        fir[:,:,:delay] = 0
        # win = np.hanning(fir.shape[2] * 2 + 1)[fir.shape[2]:-1]
        # fir = fir * win
        fir_t = np.arange(fir.shape[2]) / fs
        # npatch = fir.shape[0]

        self.voltage = interp1d(v_t, v)

        p0 = pressure_es(v[0], x0, g_eff)

        # self.voltage = v
        # self.voltage_t = v_t
        self.fir = fir
        self.fir_t = fir_t
        self.freqs = freqs
        self.ginv = ginv

        self.time = [0,]
        self.displacement = [x0,]
        self.pressure = [p0,]

        self.step_min = 1 / fs
        self.gap_eff = g_eff
        self.delta = []
        # self.time_index = 0

    def step(self):
        
        p = np.array(self.pressure)
        t = self.time[-1]
        fir = self.fir
        fs = 1  / self.step_min
        x = self.displacement[-1]

        t_new = t + self.step_min
        x_new = convolve_fir(fir, p, fs, offset=1)
        v_new = self.voltage(t_new)
        p_new = pressure_es(v_new, x_new, self.gap_eff)

        x_check = convolve_fir(fir, p, fs, offset=0)
        delta = np.max(np.abs(x_new - x_check)) / self.gap_eff #np.max(np.abs(x_check))
        # print(f'x[n] {x[4]}')
        # print(f'delta: {delta}')
        # print(f'x[n+1]: {x_new[4]}')
        # print(f'xc[n]: {x_check[4]}')

        self.displacement.append(x_new)
        self.pressure.append(p_new)
        self.time.append(t_new)
        self.delta.append(delta)

    def step_check(self, x_new=None, append=False):
        
        p = np.array(self.pressure).copy()
        t = self.time[-1]
        fir = self.fir
        fs = 1  / self.step_min
        x = self.displacement[-1]

        t_new = t + self.step_min
        if x_new is None:
            x_new = convolve_fir(fir, p, fs, offset=1)
        v_new = self.voltage(t_new)
        p_new = pressure_es(v_new, x_new, self.gap_eff)

        p = np.append(p, np.atleast_2d(p_new), axis=0)

        x_check = convolve_fir(fir, p, fs, offset=0)
        delta = np.max(np.abs(x_new - x_check)) / self.gap_eff #np.max(np.abs(x_check))
        if not append:
            self.step_check(x_new=x_check, append=True)

        # print(f'delta: {np.max(x_check)}')
        # print(f'x[n+1]: {x_new[4]}')
        # print(f'xc[n+1]: {x_check[4]}')

        if append:
            self.displacement.append(x_new)
            self.pressure.append(p_new)
            self.time.append(t_new)
            self.delta.append(delta)

    def variable_step(self, n):

        p = np.array(self.pressure)
        t = self.time[-1]
        x = self.displacement[-1]
        fir = self.fir
        fs = 1  / self.step_min

        t_new = t + self.step_min * n
        x_new = convolve_fir(fir, p, fs, offset=n)
        v_new = self.voltage(t_new)
        p_new = pressure_es(v_new, x_new, self.gap_eff)

        pi = interp1d([t, t_new], [p[-1,:], p_new], axis=0)
        xi = interp1d([t, t_new], [x, x_new], axis=0)

        for i in range(n):
            tt = t + self.step_min * (i + 1)

            self.displacement.append(xi(tt))
            self.pressure.append(pi(tt))
            self.time.append(tt)


def gausspulse(fc, fbw, fs, tpr=-100):
    cutoff = sp.signal.gausspulse('cutoff', fc=fc, bw=fbw, tpr=tpr, bwr=-3)
    adj_cutoff = np.ceil(cutoff * fs) / fs

    t = np.arange(-adj_cutoff, adj_cutoff + 1 / fs, 1 / fs)
    pulse, quad = sp.signal.gausspulse(t, fc=fc, bw=fbw, retquad=True, bwr=-3)

    return t, pulse


tstart = 0
tstop = 10e-6
g_eff = 100e-9
v_t = np.arange(0, 10e-6, 5e-9)
vdc = 1 / (1 + np.exp(-25e6 * (v_t - 2e-6)))
# v = np.zeros(len(v_t))
# vdc = 20 * np.ones(len(t))
# t, vac = gausspulse(5e6, 0.8, 1 / (5e-9))
# vac = np.roll(np.pad(vac, ((0, len(vdc) - len(vac))), mode='constant'), 600)
# v = vdc + vac
vac = np.sin(2 * np.pi * 1e6 * v_t)
# v = 40 * gausspulse(7e6, 1, fs=fs)[:len(t)]
# v = 20 * sp.signal.square(2 * np.pi * 1e6 * t)
# v = np.pad(v, (20, 0), mode='constant')[:-20]
# v = 20 * vdc + 5 * vac
# v = 5 * vac
v = 30 * vdc

x0 = np.zeros(9)

solver1 = Solver(v, v_t, x0, g_eff, delay=0)
solver2 = Solver(v, v_t, x0, g_eff, delay=0)

for i in range(600):
    solver1.step()

for i in range(600):
    solver2.step_check()



fig, ax = plt.subplots()
ax.plot(solver1.time, np.array(solver1.displacement)[:,4] / 1e-9, '.-')
ax.plot(solver2.time, np.array(solver2.displacement)[:,4] / 1e-9, '.-')
tax = ax.twinx()
tax.plot(v_t, v, '--', color='orange')
ax.set_title('Displacement')
fig.show()

fig, ax = plt.subplots()
ax.plot(solver1.time, np.array(solver1.pressure)[:,4] / 1e-9, '.-')
ax.plot(solver2.time, np.array(solver2.pressure)[:,4] / 1e-9, '.-')
tax = ax.twinx()
tax.plot(v_t, v, '--', color='orange')
ax.set_title('Pressure')
fig.show()

fig, ax = plt.subplots()
ax.plot(np.array(solver1.delta), '.-')
ax.plot(np.array(solver2.delta), '.-')
fig.show()

# fir = solver1.fir
# freqs = solver1.freqs
# ginv = solver1.ginv
# for i in tqdm(range(len(v) - 1)):
#     solver.step()


# freqs, ginv = impulse_response.read_freq_resp_db('freq_resp.db')
# win = np.hanning(ginv.shape[2] * 2 + 1)[ginv.shape[2]:-1]
# win = sp.signal.tukey(ginv.shape[2] * 2 + 1, alpha=0.1)[ginv.shape[2]:-1]
# ginv = ginv * win
# freqs2, ginv2 = one_to_two(freqs, np.conj(ginv))
# fs = (freqs2[1] - freqs2[0]) * len(freqs2)
# fir = np.real(ifft(np.conj(ginv2), axis=-1)) * fs

# G = ginv2[4,4,:]
# C = np.real(G) - 1j * np.imag(sp.signal.hilbert(np.real(G)))
# nfft = len(freqs2)
# cfir = np.real(ifft(C)) * fs
# fir = np.real(ifft(G)) * fs

# plt.figure()
# plt.plot(freqs2[:nfft // 2], np.unwrap(np.angle(C[:nfft // 2])))
# plt.plot(freqs2[:nfft // 2], np.unwrap(np.angle(G[:nfft // 2])))

# plt.figure()
# plt.plot(cfir)
# plt.plot(fir)
# plt.show()


# fig, ax = plt.subplots()
# ax.plot(t, solver.x[4,:] / 1e-9)
# tax = ax.twinx()
# tax.plot(t, v, '--', color='orange')
# # plt.plot(t, solver.x[13,:])
# fig.show()

# fig, ax = plt.subplots()
# ax.plot(t[:-1], solver.f[4,:-1])
# tax = ax.twinx()
# tax.plot(t, v, '--', color='orange')
# # plt.plot(t, solver.x[13,:])
# fig.show()