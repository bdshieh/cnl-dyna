''' Generates patch-to-patch impulse responses (in frequency domain) database for an array of CMUT membranes.
'''
import numpy as np
import multiprocessing
from itertools import repeat
from tqdm import tqdm
import os, sys, traceback

from cnld import abstract, bem, util
from cnld.mesh import Mesh, calc_mesh_refn_square
from cnld.impulse_response import create_database, update_database


## PROCESS FUNCTIONS ##

def init_process(_write_lock):
    global write_lock
    write_lock = _write_lock


def process(job):
    ''''''
    job_id, (cfg, args, f, k) = job

    # get options and parameters
    f = f[0] # remove enclosing list
    k = k[0]
    c = cfg.sound_speed
    array = abstract.load(cfg.array_config)
    firstmem = array.elements[0].membranes[0]
    file = args.file

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
    hmkwrds = ['aprx', 'basis', 'admis', 'eta', 'eps', 'm', 'clf', 'eps_aca', 'rk', 'q_reg', 'q_sing', 'strict']
    hmargs = { k:cfg[k] for k in hmkwrds }
    Z = bem.Z_matrix('HFormat', mesh, k, **hmargs)

    # construct G in HFormat from MBK and Z
    MBK.to_hformat(Z).add(Z)
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

        # write results to database
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

def main(cfg, args):
    ''''''
    # get parameters from config and args
    file = args.file
    write_over = args.write_over
    threads = args.threads if args.threads else multiprocessing.cpu_count()
    f_start, f_stop, f_step = cfg.freqs
    c = cfg.sound_speed

    # calculate job-related values
    freqs = np.arange(f_start, f_stop + f_step, f_step)
    wavenums = 2 * np.pi * freqs / c
    is_complete = None
    njobs = len(freqs)
    ijob = 0

    # check for existing file
    if os.path.isfile(file):
        if write_over:  # if file exists, write over
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

    # start multiprocessing pool and run process
    try:
        write_lock = multiprocessing.Lock()
        pool = multiprocessing.Pool(threads, initializer=init_process, initargs=(write_lock,))
        jobs = util.create_jobs(cfg, args, (freqs, 1), (wavenums, 1), mode='zip', is_complete=is_complete)
        result = pool.imap_unordered(run_process, jobs)
        for r in tqdm(result, desc='Calculating', total=njobs, initial=ijob):
            pass
    except Exception as e:
        print(e)
    finally:
        pool.terminate()
        pool.close()


if __name__ == '__main__':

    import sys
    from cnld import util

    # define configuration for this script
    Config = {}
    Config['freqs'] = 500e3, 10e6, 500e3
    Config['sound_speed'] = 1500.
    Config['array_config'] = ''
    Config['kmat_file'] = ''
    Config['aprx'] = 'paca'
    Config['basis'] = 'linear'
    Config['admis'] = 'max'
    Config['eta'] = 1.1
    Config['eps'] = 1e-12
    Config['m'] = 4
    Config['clf'] = 16
    Config['eps_aca'] = 1e-2
    Config['rk'] = 0
    Config['q_reg'] = 2
    Config['q_sing'] = 4
    Config['strict'] = False

    # get script parser and parse arguments
    parser = util.script_parser(main, Config)
    args = parser.parse_args()
    args.func(args)


