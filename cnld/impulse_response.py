'''
'''
import numpy as np
import sqlite3 as sql
import pandas as pd
from scipy.fftpack import fft, ifft, fftshift, ifftshift, fftfreq

from cnld import util

# register adapters for sqlite to convert numpy types
sql.register_adapter(np.float64, float)
sql.register_adapter(np.float32, float)
sql.register_adapter(np.int64, int)
sql.register_adapter(np.int32, int)
sql.register_adapter(np.uint64, int)
sql.register_adapter(np.uint32, int)


@util.open_db
def create_db(con, **kwargs):
    with con:
        create_displacements_table(con, **kwargs)


@util.open_db
def update_db(con, **kwargs):
    '''''''
    row_keys = ['source_patch', 'dest_patch', 'time', 'displacement']
    row_data = tuple([kwargs[k] for k in row_keys])

    with con:
        query = 'INSERT INTO displacements VALUES (?, ?, ?, ?)'
        con.executemany(query, zip(*row_data))


@util.open_db
def read_db(con):
    with con:
        query = '''
                SELECT source_patch, dest_patch, time, displacement FROM displacements
                ORDER BY source_patch, dest_patch, time
                '''
        table = pd.read_sql(query, con)
        
    source_patches = np.unique(table['source_patch'].values)
    dest_patches = np.unique(table['dest_patch'].values)
    times = np.unique(table['time'].values)
    nsource = len(source_patches)
    ndest = len(dest_patches)
    ntime = len(times)

    disp = np.array(table['displacement']).reshape((nsource, ndest, ntime))
    return times, disp


@util.open_db
def create_displacements_table(con, **kwargs):
    ''''''
    with con:
        query = '''
                CREATE TABLE displacements (
                source_patch integer,
                dest_patch integer,
                time float,
                displacement float
                )
                '''
        con.execute(query)

        # create indexes
        con.execute('CREATE INDEX time_index ON displacements (time)')
        con.execute('CREATE INDEX source_patch_index ON displacements (source_patch)')
        con.execute('CREATE INDEX dest_patch_index ON displacements (dest_patch)')


def one_to_two(f, s, axis=-1, odd=False):
    '''
    Converts a one-sided FFT (of a real-valued signal) to a two-sided FFT.
    Energy is not halved by this function.
    '''
    s = np.atleast_2d(s)

    # determine two-sided signal length
    nf = s.shape[axis]
    nfft = 2 * nf - 1 if odd else nfft = (nf - 1) * 2

    # create empty spectrum
    newshape = list(s.shape)
    newshape[axis] = nfft
    s2s = np.zeros(newshape, dtype=np.complex128)

    # construct positive frequencies
    idx1 = [slice(None)] * s.ndim
    idx2 = [slice(None)] * s.ndim
    if odd:
        idx1[axis] = slice(None, nfft // 2 + 1)
        idx2[axis] = slice(None, None, None)
    else:
        idx1[axis] = slice(None, nfft // 2)
        idx2[axis] = slice(None, -1, None)
    s2s[tuple(idx1)] = s[tuple(idx2)]

    # construct negative frequencies
    idx1 = [slice(None)] * s.ndim
    idx2 = [slice(None)] * s.ndim
    if odd:
        idx1[axis] = slice(nfft // 2 + 1, None, None)
        idx2[axis] = slice(-1, 0, -1)
    else:
        idx1[axis] = slice(nfft // 2, None, None)
        idx2[axis] = slice(-1, 0, -1)
    s2s[tuple(idx1)] = np.conj(s[tuple(idx2)])

    # create vector of new frequency bins
    df = f[1] - f[0]
    fs = df * (nfft // 2)
    f2s = fftfreq(nfft, 1 / fs)

    return f2s, s2s


def two_to_one(f, s, axis=-1):
    '''
    Converts a two-sided FFT (of a real-valued signal) to a one-sided FFT.
    Energy is not doubled by this function.
    '''
    s = np.atleast_2d(s)

    # determine one-sided signal length
    nf = s.shape[axis]
    nfft = (nf + 1) // 2 if nf % 2 == 0 else (nf // 2) + 1

    # index into spectrum
    idx = [slice(None)] * s.ndim
    idx[axis] = slice(None, nfft, None)
    s1s = s[idx].copy()

    # create vector of new frequency bins
    df = f[1] - f[0]
    fs = df * nfft
    f1s = fftfreq(nfft, 1 / fs)

    return f1s, s1s


def kramers_kronig(s, axis=-1):
    '''
    Reconstructs spectrum with phase based on the Kramers-Kronig relations
    for a causal LTI system.
    '''
    return np.real(s) - 1j * np.imag(sp.signal.hilbert(np.real(s), axis=axis))


def fft_to_fir(f, s, axis=-1)
    '''
    Convert a one-sided FFT to an impulse response representing a causaul LTI system.
    '''
    f2s, s2s = one_to_two(f, kramers_kronig(s, axis=axis), axis=axis)

    df = f[1] - f[0]
    fs = df * nfft
    t = np.arange(nfft) / fs
    return t, np.real(ifft(s, axis=axis)) * fs


if __name__ == '__main__':
    pass


