
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

from cnld import abstract, bem, util
from cnld.mesh import Mesh, calc_mesh_refn_square
from cnld.impulse_response import create_database, update_database


defaults = dict()
defaults['threads'] = multiprocessing.cpu_count()

hmopts = {}
hmopts['aprx'] = 'paca'
hmopts['basis'] = 'linear'
hmopts['admis'] = 'max'
hmopts['eta'] = 1.1
hmopts['eps'] = 1e-12
hmopts['m'] = 4
hmopts['clf'] = 16
hmopts['eps_aca'] = 1e-2
hmopts['rk'] = 0
hmopts['q_reg'] = 2
hmopts['q_sing'] = 4
hmopts['strict'] = False

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

    job_id, (file, f, k, simopts, array) = job

    # get options and parameters
    f = f[0] # remove enclosing list
    k = k[0]
    c = simopts.sound_speed
    firstmem = array.elements[0].membranes[0]

    # determine mesh refn needed based on first membrane
    wavelen = 2 * np.pi * f / c
    length_x = firstmem.length_x
    length_y = firstmem.length_y
    refn = calc_mesh_refn_square(length_x, length_y, wavelen)
    
    # create mesh
    mesh = Mesh.from_abstract(array, refn)

    # create MBK matrix in SparseFormat based on first membrane
    n = len(mesh.vertices)
    nmem = abstract.get_membrane_count(array)
    rho = firstmem.rho
    h = firstmem.h
    att = firstmem.att
    kfile = firstmem.k_matrix_comsol_file
    MBK = bem.MBK_matrix(f, len(mesh.vertices), nmem, rho, h, att, kfile, compress=True)

    # create Z matrix in HFormat
    Z = bem.Z_matrix('HFormat', mesh, k, **hmopts)

    # construct G in HFormat from MBK and Z
    MBK.to_hformat().add(Z)
    G = MBK

    # LU decomposition
    G_LU = G.lu()

    # solve for patch to patch responses
    npatch = abstract.get_patch_count(array)
    source_patch_id = np.arange(npatch)
    dest_patch_id = np.arange(npatch)

    for sid in source_patch_id:
        
        # solve
        b = np.zeros(len(mesh.vertices))
        mask = np.any(mesh.patch_ids == sid, axis=1)
        b[mask] = 1
        x = G_LU.lusolve(b)

        x_patch = []
        for did in dest_patch_id:

            mask = np.any(mesh.patch_ids == did)
            x_patch.append(np.mean(x[mask]))

        data = {}
        data['frequency'] = repeat(f)
        data['wavenumber'] = repeat(k)
        data['source_patch_id'] = repeat(sid)
        data['dest_patch_id'] = dest_patch_id
        data['displacement_real'] = np.real(x_patch)
        data['displacement_imag'] = np.imag(x_patch)

        with write_lock:
            update_database(file, **data)
            util.update_progress(file, job_id)


def run_process(*args, **kwargs):

    try:
        return process(*args, **kwargs)
    except:
        raise Exception("".join(traceback.format_exception(*sys.exc_info())))


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


