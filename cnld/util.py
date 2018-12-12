'''
Utility functions.
''' 

import numpy as np
import pandas as pd
import scipy as sp
import scipy.signal
import scipy.fftpack
from scipy.spatial.distance import cdist
import itertools
from contextlib import closing
from itertools import repeat
import sqlite3 as sql
import argparse
from cnld import abstract


''' GEOMETRY-RELATED FUNCTIONS '''

def meshview(v1, v2, v3, mode='cartesian', as_list=True):

    if mode.lower() in ('cart', 'cartesian'):
        x, y, z = np.meshgrid(v1, v2, v3, indexing='ij')

    elif mode.lower() in ('sph', 'spherical'):
        r, theta, phi = np.meshgrid(v1, np.deg2rad(v2), np.deg2rad(v3), indexing='ij')
        x, y, z = sph2cart(r, theta, phi)

    elif mode.lower() in ('sec', 'sector'):
        r, alpha, beta = np.meshgrid(v1, np.deg2rad(v2), np.deg2rad(v3), indexing='ij')
        x, y, z = sec2cart(r, alpha, beta)

    elif mode.lower() in ('dp', 'dpolar'):
        r, alpha, beta = np.meshgrid(v1, np.deg2rad(v2), np.deg2rad(v3), indexing='ij')
        x, y, z = dp2cart(r, alpha, beta)

    if as_list:
        return np.c_[x.ravel('F'), y.ravel('F'), z.ravel('F')]
    else:
        return x, y, z


def sec2cart(r, alpha, beta):

    z = r / np.sqrt(np.tan(alpha)**2 + np.tan(beta)**2 + 1)
    x = z * np.tan(alpha)
    y = z * np.tan(beta)

    # alpha_p = np.arctan(np.tan(alpha) * np.cos(beta))
    # x = np.sin(alpha_p) * r
    # y = -np.sin(beta) * r * np.cos(alpha_p)
    # z = np.sqrt(r**2 - x**2 - y**2)

    # px = -px
    # pyp = np.arctan(np.cos(px) * np.sin(py) / np.cos(py))
    # x = r * np.sin(pyp)
    # y = -r * np.cos(pyp) * np.sin(px)
    # z = r * np.cos(px) * np.cos(pyp)

    return x, y, z


def cart2sec(x, y, z):

    r = np.sqrt(x**2 + y**2 + z**2)
    alpha = np.arccos(z / (np.sqrt(x**2 + z**2))) * np.sign(x)
    beta = np.arccos(z / (np.sqrt(y**2 + z**2))) * np.sign(y)

    # r = np.sqrt(x**2 + y**2 + z**2)
    # alpha_p = np.arcsin(x / r)
    # beta = -np.arcsin(-y / r / np.cos(alpha_p))
    # alpha = np.arctan(np.tan(alpha_p) / np.cos(beta))

    return r, alpha, beta


def sph2cart(r, theta, phi):

    x = r * np.cos(theta) * np.sin(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(phi)

    return x, y, z


def cart2sph(x, y, z):

    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan(y / x)
    phi = np.arccos(z / r)

    return r, theta, phi


def cart2dp(x, y, z):

    r = np.sqrt(x**2 + y**2 + z**2)
    alpha = np.arccos((np.sqrt(y**2 + z**2) / r))
    beta = np.arccos((np.sqrt(x**2 + z**2) / r))

    return r, alpha, beta


def dp2cart(r, alpha, beta):

    z = r * (1 - np.sin(alpha)**2 - np.sin(beta)**2)
    x = r * np.sin(alpha)
    y = r * np.sin(beta)

    return x, y, z


def rotation_matrix(vec, angle):

    if isinstance(vec, str):
        string = vec.lower()
        if string == 'x':
            vec = [1, 0, 0]
        elif string == '-x':
            vec = [-1, 0, 0]
        elif string == 'y':
            vec = [0, 1, 0]
        elif string == '-y':
            vec = [0, -1, 0]
        elif string == 'z':
            vec = [0, 0, 1]
        elif string == '-x':
            vec = [0, 0, -1]

    x, y, z = vec
    a = angle

    r = np.zeros((3, 3))
    r[0, 0] = np.cos(a) + x**2 * (1 - np.cos(a))
    r[0, 1] = x * y * (1 - np.cos(a)) - z * np.sin(a)
    r[0, 2] = x * z * (1 - np.cos(a)) + y * np.sin(a)
    r[1, 0] = y * x * (1 - np.cos(a)) + z * np.sin(a)
    r[1, 1] = np.cos(a) + y**2 * (1 - np.cos(a))
    r[1, 2] = y * z * (1 - np.cos(a)) - x * np.sin(a)
    r[2, 0] = z * x * (1 - np.cos(a)) - z * np.sin(a)
    r[2, 1] = z * y * (1 - np.cos(a)) + x * np.sin(a)
    r[2, 2] = np.cos(a) + z**2 * (1 - np.cos(a))

    return r


def rotate_nodes(nodes, vec, angle):

    rmatrix = rotation_matrix(vec, angle)
    return rmatrix.dot(nodes.T).T


def distance(*args):
    return cdist(*np.atleast_2d(*args))


''' SIGNAL PROCESSING AND RF DATA FUNCTIONS '''

def gausspulse(fc, fbw, fs):

    cutoff = scipy.signal.gausspulse('cutoff', fc=fc, bw=fbw, tpr=-100, bwr=-3)
    adj_cutoff = np.ceil(cutoff * fs) / fs

    t = np.arange(-adj_cutoff, adj_cutoff + 1 / fs, 1 / fs)
    pulse, _ = sp.signal.gausspulse(t, fc=fc, bw=fbw, retquad=True, bwr=-3)

    return pulse, t


def nextpow2(n):
    return 2 ** int(np.ceil(np.log2(n)))


def envelope(rf_data, N=None, axis=-1):
    return np.abs(scipy.signal.hilbert(np.atleast_2d(rf_data), N, axis=axis))


def qbutter(x, fn, fs=1, btype='lowpass', n=4, plot=False, axis=-1):

    wn = fn / (fs / 2.)
    b, a = sp.signal.butter(n, wn, btype)

    # if plot:
    #     w, h = freqz(b, a)
    #
    #     fig = plt.figure()
    #     ax1 = fig.add_subplot(111)
    #
    #     ax1.plot(w / (2 * np.pi) * fs, 20 * np.log10(np.abs(h)), 'b')
    #     ax1.set_xlabel('Frequency (Hz)')
    #     ax1.set_ylabel('Amplitude (dB)')
    #     ax1.set_title('Butterworth filter response')
    #
    #     ax2 = ax1.twinx()
    #     ax2.plot(w / (2 * np.pi) * fs, np.unwrap(np.angle(h)), 'g')
    #     ax2.set_ylabel('Phase (radians)')
    #
    #     plt.grid()
    #     plt.axis('tight')
    #
    #     fig.show()

    fx = sp.signal.lfilter(b, a, x, axis=axis)

    return fx


def qfirwin(x, fn, fs=1, btype='lowpass', ntaps=80, plot=False, axis=-1,
            window='hamming'):
    if btype.lower() in ('lowpass', 'low'):
        pass_zero = 1
    elif btype.lower() in ('bandpass', 'band'):
        pass_zero = 0
    elif btype.lower() in ('highpass', 'high'):
        pass_zero = 0

    wn = fn / (fs / 2.)
    b = sp.signal.firwin(ntaps, wn, pass_zero=pass_zero, window=window)

    # if plot:
    #     w, h = freqz(b)
    #
    #     fig = plt.figure()
    #     ax1 = fig.add_subplot(111)
    #
    #     ax1.plot(w / (2 * np.pi) * fs, 20 * np.log10(np.abs(h)), 'b')
    #     ax1.set_xlabel('Frequency (Hz)')
    #     ax1.set_ylabel('Amplitude (dB)')
    #     ax1.set_title('FIR filter response')
    #
    #     ax2 = ax1.twinx()
    #     ax2.plot(w / (2 * np.pi) * fs, np.unwrap(np.angle(h)), 'g')
    #     ax2.set_ylabel('Phase (radians)')
    #
    #     plt.grid()
    #     plt.axis('tight')
    #
    #     fig.show()

    fx = np.apply_along_axis(lambda x: np.convolve(x, b), axis, x)

    return fx


def qfft(s, nfft=None, fs=1, dr=100, fig=None, **kwargs):
    '''
    Quick FFT plot. Returns frequency bins and FFT in dB.
    '''
    s = np.atleast_2d(s)

    nsig, nsample = s.shape

    if nfft is None:
        nfft = nsample

    # if fig is None:
    #     fig = plt.figure(tight_layout=1)
    #     ax = fig.add_subplot(111)
    # else:
    #     ax = fig.get_axes()[0]

    if nfft > nsample:
        s = np.pad(s, ((0, 0), (0, nfft - nsample)), mode='constant')
    elif nfft < nsample:
        s = s[:, :nfft]

    ft = sp.fftpack.fft(s, axis=1)
    freqs = sp.fftpack.fftfreq(nfft, 1 / fs)

    ftdb = 20 * np.log10(np.abs(ft) / (np.max(np.abs(ft), axis=1)[..., None]))
    ftdb[ftdb < -dr] = -dr

    cutoff = (nfft + 1) // 2

    # ax.plot(freqs[:cutoff], ftdb[:, :cutoff].T, **kwargs)
    # ax.set_xlabel('Frequency (Hz)')
    # ax.set_ylabel('Magnitude (dB re max)')
    # fig.show()

    return freqs[:cutoff], ftdb[:, :cutoff]


''' JOB-RELATED FUNCTIONS '''

def chunks(iterable, n):

    res = []
    for el in iterable:
        res.append(el)
        if len(res) == n:
            yield res
            res = []
    if res:
        yield res


def create_jobs(*args, mode='zip', is_complete=None):
    '''
    Convenience function for creating jobs (sets of input arguments) for multiprocessing Pool. Supports zip and product
    combinations, and automatic chunking of iterables.
    '''
    static_args = list()
    static_idx = list()
    iterable_args = list()
    iterable_idx = list()

    for arg_no, arg in enumerate(args):
        if isinstance(arg, (tuple, list)):
            iterable, chunksize = arg
            if chunksize == 1:
                iterable_args.append(iterable)
            else:
                iterable_args.append(chunks(iterable, chunksize))
            iterable_idx.append(arg_no)
        else:
            static_args.append(itertools.repeat(arg))
            static_idx.append(arg_no)

    if not iterable_args and not static_args:
        return

    if not iterable_args:
        yield 1, tuple(args[i] for i in static_idx)

    if not static_args:
        repeats = itertools.repeat(())
    else:
        repeats = zip(*static_args)

    if mode.lower() == 'product':
        combos = itertools.product(*iterable_args)
    elif mode.lower() == 'zip':
        combos = zip(*iterable_args)
    elif mode.lower() == 'zip_longest':
        combos = itertools.zip_longest(*iterable_args)

    for job_id, (r, p) in enumerate(zip(repeats, combos)):
        # skip jobs that have been completed
        if is_complete is not None and is_complete[job_id]:
            continue

        res = r + p
        # reorder vals according to input order
        yield job_id + 1, tuple(res[i] for i in np.argsort(static_idx + iterable_idx))


''' DATABASE FUNCTIONS '''

def open_db(f):
    def decorator(firstarg, *args, **kwargs):
        if isinstance(firstarg, sql.Connection):
            return f(firstarg, *args, **kwargs)
        else:
            with closing(sql.connect(firstarg)) as con:
                return f(con, *args, **kwargs)

    return decorator


@open_db
def table_exists(con, name):

    query = '''SELECT count(*) FROM sqlite_master WHERE type='table' and name=?'''
    return con.execute(query, (name,)).fetchone()[0] != 0


@open_db
def create_metadata_table(con, **kwargs):

    table = [[str(v) for v in list(kwargs.values())]]
    columns = list(kwargs.keys())
    pd.DataFrame(table, columns=columns, dtype=str).to_sql('metadata', con, if_exists='replace', index=False)


@open_db
def create_progress_table(con, njobs):

    with con:
        # create table
        con.execute('CREATE TABLE progress (job_id INTEGER PRIMARY KEY, is_complete boolean)')
        # insert values
        con.executemany('INSERT INTO progress (is_complete) VALUES (?)', repeat((False,), njobs))


@open_db
def get_progress(con):

    table = pd.read_sql('SELECT is_complete FROM progress ORDER BY job_id', con)

    is_complete = np.array(table).squeeze()
    ijob = sum(is_complete) + 1

    return is_complete, ijob


@open_db
def update_progress(con, job_id):

    with con:
        con.execute('UPDATE progress SET is_complete=1 WHERE job_id=?', [job_id,])


''' SCRIPTING FUNCTIONS '''

def script_parser(main, config_def):
    '''
    General script command-line interface with 'config' and 'run' subcommands.
    '''
    if isinstance(config_def, dict):
        # create config abstract type based on supplied dict
        Config = abstract.register_type('Config', config_def)
    else:
        # config abstract type already defined
        Config = config_def

    # config subcommand generates a default configuration template
    def config(args):
        abstract.dump(Config(), args.file)

    # run subcommand will load the config file and pass to main
    def run(args):
        if args.config:
            cfg = Config(**abstract.load(args.config))
        else:
            cfg = Config()
        return main(cfg, args)

    # create argument parser
    parser = argparse.ArgumentParser()
    # define config subparser
    subparsers = parser.add_subparsers(help='sub-command help')
    config_parser = subparsers.add_parser('config', help='config_help')
    config_parser.add_argument('file')
    config_parser.set_defaults(func=config)
    # define run subparser
    run_parser = subparsers.add_parser('run', help='run_help')
    run_parser.add_argument('config', nargs='?')
    run_parser.add_argument('-f', '--file', nargs='?')
    run_parser.add_argument('-t', '--threads', nargs='?', type=int)
    run_parser.add_argument('-w', '--write-over', action='store_true')
    run_parser.set_defaults(func=run)

    return parser, run_parser


''' MISC FUNCTIONS '''

def memoize(func):
    '''
    Simple memoizer to cache repeated function calls.
    '''
    def ishashable(obj):
        try:
            hash(obj)
        except TypeError:
            return False
        return True
    
    def make_hashable(obj):
        if not ishashable(obj):
            return str(obj)
        return obj

    memo = {}
    def decorator(*args):
        key = tuple(make_hashable(a) for a in args)
        if key not in memo:
            memo[key] = func(*args)
        return memo[key]
    return decorator