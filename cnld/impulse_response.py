'''
'''
import numpy as np
import sqlite3 as sql
import pandas as pd
from scipy.signal import hilbert
from scipy.fftpack import fft, ifft, fftshift, ifftshift, fftfreq
from scipy.interpolate import interp1d

from cnld import util


def one_to_two(f, s, axis=-1, odd=False):
    '''
    Converts a one-sided FFT (of a real-valued signal) to a two-sided FFT.
    Energy is not halved by this function.
    '''
    def func1d(s):

        # create empty spectrum
        s2s = np.zeros(nfft, dtype=np.complex128)

        # construct positive frequencies
        if odd:
            s2s[:nfft // 2 + 1] = s[:]
        else:
            s2s[:nfft // 2] = s[:-1:]
        
        # construct negative frequencies
        if odd:
            s2s[nfft // 2 + 1::] = np.conj(s[-1:0:-1])
        else:
            s2s[nfft // 2::] = np.conj(s[-1:0:-1])
        
        return s2s
    
    # determine two-sided signal length
    nf = s.shape[axis]
    nfft = 2 * nf - 1 if odd else (nf - 1) * 2

    s2s = np.apply_along_axis(func1d, axis, s)

    # create vector of new frequency bins
    df = f[1] - f[0]
    fs = df * nfft
    f2s = fftfreq(nfft, 1 / fs)

    return f2s, s2s


def two_to_one(f, s, axis=-1):
    '''
    Converts a two-sided FFT (of a real-valued signal) to a one-sided FFT.
    Energy is not doubled by this function.
    '''
    def func1d(s):
        # index into spectrum
        s1s = s[:nfft:].copy()
        return s1s

    # determine one-sided signal length
    nf = s.shape[axis]
    nfft = (nf + 1) // 2 if nf % 2 == 0 else (nf // 2) + 1

    s1s = np.apply_along_axis(func1d, axis, s)

    # create vector of new frequency bins
    df = f[1] - f[0]
    fs = df * nfft * 2
    f1s = fftfreq(nfft, 1 / fs)

    return f1s, s1s


def kramers_kronig(f, s, axis=-1):
    '''
    Reconstructs spectrum with phase based on the Kramers-Kronig relations
    for a causal LTI system.
    '''
    def func1d(s):
        # mag = np.abs(s)
        phase = np.angle(s)

        # unwrap phase and estimate front delay
        fd = np.unwrap(phase) / _omg
        fd[np.isnan(fd)] = np.inf
        tmin = np.round(np.min(fd) * fs) / fs
        
        # remove front delay
        h = s * np.exp(-1j * omg * tmin)
        mag = np.abs(h)

        # calculate phase from log magnitude
        if mag[0] == 0:
            a = -np.log(mag[1:])
            phi = -np.imag(hilbert(a))
            kkr = np.exp(-a - 1j * phi)
            kkr = np.insert(kkr, 0, h[0])
        else:
            a = -np.log(mag)
            phi = -np.imag(hilbert(a))
            kkr = np.exp(-a - 1j * phi)

        kkr *= np.exp(1j * omg * tmin)
        return kkr

    omg = 2 * np.pi * f
    _omg = omg.copy()
    _omg[_omg == 0] = np.nan

    df = f[1] - f[0]
    fs = df * len(f) 
    
    return np.apply_along_axis(func1d, axis, s)


def interp_fft(f, s, mult, axis=-1):

    if mult == 1:
        return f, s

    fi = np.linspace(f[0], f[-1], (len(f) - 1) * mult + 1)

    mag = np.abs(s)
    phase = np.unwrap(np.angle(s), axis=axis)
    imag = interp1d(f, mag, kind='linear', axis=axis)
    iphase = interp1d(f, phase, kind='linear', axis=axis)

    return fi, imag(fi) * np.exp(1j * iphase(fi))


def fft_to_fir(f, s, mult=1, use_kkr=True, axis=-1):
    '''
    Convert a one-sided FFT to an impulse response representing a delay causal LTI system.
    '''
    fi, si = interp_fft(f, s, mult=mult, axis=axis)
    f2s, s2s = one_to_two(fi, si, axis=axis, odd=True)

    if use_kkr:
        s2s = kramers_kronig(f2s, s2s, axis=axis)

    df = f2s[1] - f2s[0]
    nfft = len(f2s)
    fs = df * nfft
    t = np.arange(nfft) / fs
    return t, np.real(ifft(s2s, axis=axis)) * fs


if __name__ == '__main__':
    pass


