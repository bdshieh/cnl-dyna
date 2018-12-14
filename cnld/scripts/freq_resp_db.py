''' 
Generates patch-to-patch impulse responses (in frequency domain) database for an array of CMUT membranes.
'''
import numpy as np
import multiprocessing
from itertools import repeat
from tqdm import tqdm
import os, sys, traceback

from cnld import abstract, bem, util
from cnld.mesh import Mesh, calc_refn_square
from cnld.impulse_response import create_database, update_database


''' PROCESS FUNCTIONS '''

def init_process(_write_lock, _cfg, _args):
    global write_lock, cfg, args
    write_lock = _write_lock
    cfg = _cfg
    args = _args


def process(job):
    ''''''
    job_id, (f, k) = job

    # get options and parameters
    c = cfg.sound_speed
    array = abstract.load(cfg.array_config)
    firstmem = array.elements[0].membranes[0]
    file = args.file

    # determine mesh refn needed based on first membrane
    wavelen = c / f
    length_x = firstmem.length_x
    length_y = firstmem.length_y
    refn = calc_refn_square(length_x, length_y, wavelen)
    if refn < 3: refn = 3  # enforce minimum mesh refinement

    # create mesh
    mesh = Mesh.from_abstract(array, refn)

    # create MBK matrix in SparseFormat
    # MBK = bem.mbk_from_abstract(array, f, refn, format='SparseFormat')
    MBK = bem.mbk_from_abstract(array, f, refn, format='FullFormat')

    # create Z matrix in HFormat
    hmkwrds = ['aprx', 'basis', 'admis', 'eta', 'eps', 'm', 'clf', 'eps_aca', 'rk', 'q_reg', 'q_sing', 'strict']
    hmargs = { k:getattr(cfg, k) for k in hmkwrds }
    Z = bem.z_from_abstract(array, k, refn, **hmargs)

    # construct G in HFormat from MBK and Z
    # G = MBK + (2 * np.pi * f) ** 2 * -1000. * Z 
    # G = MBK + -(2 * np.pi * f)**2 * 1000. / (mesh.g[0]) * Z
    # G = MBK + -(2 * np.pi * f)**2 * 1000. * Z
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
        smask = np.any(mesh.patch_ids == sid, axis=1)
        b[smask] = 1
        x = G_LU.lusolve(b)
        x_patch = []
        dest_membrane_ids = []
        dest_element_ids = []
        for did in dest_patch_id:
            dmask = np.any(mesh.patch_ids == did, axis=1)
            x_patch.append(np.mean(x[dmask]))
            dest_membrane_ids.append(mesh.membrane_ids[dmask][0])
            dest_element_ids.append(mesh.element_ids[dmask][0])

        # write results to database
        data = {}
        data['frequency'] = repeat(f)
        data['wavenumber'] = repeat(k)
        data['source_patch_id'] = repeat(sid)
        data['dest_patch_id'] = dest_patch_id
        data['source_membrane_id'] = repeat(mesh.membrane_ids[smask][0])
        data['dest_membrane_id'] = dest_membrane_ids
        data['source_element_id'] = repeat(mesh.element_ids[smask][0])
        data['dest_element_id'] = dest_element_ids
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


''' ENTRY POINT '''

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
            create_database(file, frequencies=freqs, wavenumbers=wavenums)  # create database
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
        create_database(file, frequencies=freqs, wavenumbers=wavenums)  # create database
        util.create_progress_table(file, njobs)

    # start multiprocessing pool and run process
    try:
        write_lock = multiprocessing.Lock()
        pool = multiprocessing.Pool(threads, initializer=init_process, initargs=(write_lock, cfg, args))
        jobs = util.create_jobs((freqs, 1), (wavenums, 1), mode='zip', is_complete=is_complete)
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

    # define default configuration for this script
    Config = {}
    Config['freqs'] = 500e3, 10e6, 500e3
    Config['sound_speed'] = 1500.
    Config['array_config'] = ''
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
    parser, run_parser = util.script_parser(main, Config)
    args = parser.parse_args()
    args.func(args)


