
import numpy as np
import multiprocessing
import os
import sqlite3 as sql
from itertools import repeat
from contextlib import closing
from tqdm import tqdm
import traceback
import sys
import argparse

from cmut_nonlinear_sim import bem, util
from cmut_nonlinear_sim.impulse_response import create_database, update_database


defaults = dict()
defaults['threads'] = multiprocessing.cpu_count()

# for frequency each f, create mesh from spec

# from mesh, generate M, B, K as Scipy sparse matrices
# convert M, B, K to MBK in compressed sparseformat

# generate Z in compressed hformat
# perform G = MBK + Z by converting MBK to hformat and adding

# decompose LU = G
# solve x(f) using LU for lumped forcings
# calculate time impulse response using ifft


## PROCESS FUNCTIONS ##

def init_process(_write_lock):

    global write_lock
    write_lock = _write_lock


def process(job):

    job_id, (file, f, k, array) = job

    # remove enclosing lists
    f = f[0]

    c = 1500.
    k = 2 * np.pi * f / c


    MBK = MBK_matrix(f, n, nmem, rho, h, att, kfile, compress=True)
    Z = Z_matrix(format, mesh, k, **hm_opts)

    MBK.to_hformat().add(Z)
    G = MBK

    LU = G.lu()

    b = None
    x = LU.lusolve(b)

    data = {}
    data['displacement'] = x

    with write_lock:
        update_database(file, **data)
        util.update_progress(file, job_id)


def run_process(*args, **kwargs):

    try:
        return process(*args, **kwargs)
    except:
        raise Exception("".join(traceback.format_exception(*sys.exc_info())))


## DATABASE FUNCTIONS ##


## ENTRY POINT ##

def main():

    # define and parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('file')
    parser.add_argument('-f', '--freqs', nargs=3, type=float)
    parser.add_argument('-t', '--threads', nargs='?', type=int)
    parser.add_argument('-o', '--overwrite', action='store_true')
    args = vars(parser.parse_args())


    file = args['file']
    overwrite = args['overwrite']
    threads = args['threads'] if args['threads'] else multiprocessing.cpu_count()
    f_start, f_stop, f_step = args['freqs'] if args['freqs'] else (500e3, 10e6, 500e3)
    c = 1500.
    array = None

    freqs = np.arange(f_start, f_stop + f_step, f_step)
    wavenums = 2 * np.pi * freqs / c

    # calculate job-related values
    is_complete = None
    njobs = len(freqs)
    ijob = 0

    # check for existing file
    if os.path.isfile(file):
        if overwrite:  # if file exists, prompt for overwrite
            os.remove(file)  # remove existing file
            create_database(file, freqs=freqs, wavenums=wavenums)  # create database
            util.create_progress_table(file, njobs)

        else: # continue from current progress
            is_complete, ijob = util.get_progress(file)
            if np.all(is_complete): return

    else:
        # Make directories if they do not exist
        file_dir = os.path.dirname(os.path.abspath(file))
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        # create database
        create_database(file, freqs=freqs, wavenums=wavenums)  # create database
        util.create_progress_table(file, njobs)

    try:
        # start multiprocessing pool and run process
        write_lock = multiprocessing.Lock()
        pool = multiprocessing.Pool(threads, initializer=init_process, initargs=(write_lock,))
        jobs = util.create_jobs(file, (freqs, 1), (wavenums, 1), array, mode='zip', is_complete=is_complete)
        result = pool.imap_unordered(run_process, jobs)

        for r in tqdm(result, desc='Calculating', total=njobs, initial=ijob):
            pass

    except Exception as e:
        print(e)

    finally:

        pool.terminate()
        pool.close()


if __name__ == '__main__':
    main()


