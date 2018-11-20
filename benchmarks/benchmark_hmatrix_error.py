## benchmarks / benchmark_hmatrix_error.py ##

import numpy as np
from tqdm import tqdm
from timeit import default_timer as timer
import multiprocessing
import argparse
import sqlite3 as sql
from contextlib import closing
import traceback
import sys
import os

from cmut_nonlinear_sim.mesh import *
from cmut_nonlinear_sim.zmatrix import *
from cmut_nonlinear_sim import util


# register adapters for sqlite to convert numpy types
sql.register_adapter(np.float64, float)
sql.register_adapter(np.float32, float)
sql.register_adapter(np.int64, int)
sql.register_adapter(np.int32, int)

# hard-coded arguments for benchmark
mesh_args = {}
# mesh_args['nx'] = 12
# mesh_args['ny'] = 12
mesh_args['lengthx'] = 40e-6
mesh_args['lengthy'] = 40e-6
mesh_args['pitchx'] = 60e-6
mesh_args['pitchy'] = 60e-6

hm_args = {}
hm_args['aprx'] = 'paca'
hm_args['basis'] = 'linear'
hm_args['admis'] = 'max'
hm_args['eta'] = 1.1
hm_args['eps'] = 1e-12
hm_args['m'] = 4
hm_args['clf'] = 16
hm_args['eps_aca'] = 1e-2
hm_args['rk'] = 0
hm_args['q_reg'] = 2
hm_args['q_sing'] = 4
hm_args['strict'] = False

benchmark_args = {}
benchmark_args['eps_lu'] = 1e-12
benchmark_args['sound_speed'] = 1500.


## PROCESS FUNCTIONS ##

def nrmse(x, xhat):
    return np.sqrt(np.mean(np.abs(x - xhat) ** 2)) / np.sqrt(np.sum(np.abs(xhat) ** 2))


def benchmark(Z):

    eps_lu = benchmark_args['eps_lu']

    b = np.ones(Z.shape[0])
    
    start = timer()
    LU = lu(Z, eps=eps_lu)
    time_lu = timer() - start
    
    start = timer()
    x = lusolve(LU, b)
    time_solve = timer() - start

    results = {}
    results['x'] = x
    results['size'] = Z.size
    results['time_assemble'] = Z.assemble_time
    results['time_lu'] = time_lu
    results['time_solve'] = time_solve

    del Z

    return results


def init_process(_write_lock):

    global write_lock
    write_lock = _write_lock


def process(job):

    job_id, (file, f) = job
    f = f[0]
    
    c = benchmark_args['sound_speed']
    lengthx = mesh_args['lengthx']
    lengthy = mesh_args['lengthy']

    k = 2 * np.pi * f / c

    # determine mesh refinement needed at each frequency 
    # to maintain maximum edge size less than 10 wavelengths
    refn = 2
    wl = c / f

    while True:
        if refn > 20:
            raise Exception('Mesh refinement limit reached')

        hmax = square(lengthx, lengthy, refn=refn).hmax
        if wl / hmax > 10:
            break
        else:
            refn += 1

    mesh = fast_matrix_array(refn=refn, **mesh_args)
    
    hm = benchmark(HierarchicalMatrix(mesh, k, **hm_args))
    full = benchmark(FullMatrix(mesh, k))
    
    err = nrmse(full['x'], hm['x'])
    vertices = len(mesh.vertices)
    edges = len(mesh.edges)
    triangles = len(mesh.triangles)

    row_data = []
    row_data.append([None, f, k, vertices, edges, triangles, 'HierarchicalMatrix', hm['size'], 
        hm['time_assemble'], hm['time_lu'], hm['time_solve'], err])
    row_data.append([None, f, k, vertices, edges, triangles, 'FullMatrix', full['size'], 
        full['time_assemble'], full['time_lu'], full['time_solve'], None])

    with write_lock:
        with closing(sql.connect(file)) as con:
            update_results_table(con, row_data)
            util.update_progress(con, job_id)


def run_process(*args, **kwargs):
    try:
        return process(*args, **kwargs)
    except:
        raise Exception("".join(traceback.format_exception(*sys.exc_info())))


## DATABASE FUNCTIONS ##

def create_database(file, njobs):

    with closing(sql.connect(file)) as con:
        # create database tables (progress, results)
        util.create_progress_table(con, njobs)
        create_results_table(con)


def create_results_table(con):

    with con:
        # create table
        query = '''
                CREATE TABLE results (
                id INTEGER PRIMARY KEY,
                frequency float,
                wavenumber float,
                vertices int,
                edges int,
                triangles int,
                format str,
                size float,
                time_assemble float,
                time_lu float,
                time_solve float,
                nrmse float
                )
                '''
        con.execute(query)


def update_results_table(con, row_data):

    with con:
        query = 'INSERT INTO results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'
        con.executemany(query, row_data)


## ENTRY POINT ##

def main():

    # define and parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('file')
    parser.add_argument('-f', '--freqs', nargs=3, type=float)
    parser.add_argument('-t', '--threads', nargs='?', type=int)
    parser.add_argument('-o', '--overwrite', action='store_true')
    parser.add_argument('-n', '--nmem', nargs='?', type=int)
    args = vars(parser.parse_args())


    file = args['file']
    overwrite = args['overwrite']
    threads = args['threads'] if args['threads'] else multiprocessing.cpu_count()
    f_start, f_stop, f_step = args['freqs'] if args['freqs'] else (500e3, 10e6, 500e3)
    c = benchmark_args['sound_speed']

    nmem = args['nmem'] if args['nmem'] else 5
    mesh_args['nx'] = nmem
    mesh_args['ny'] = nmem
    
    freqs = np.arange(f_start, f_stop + f_step, f_step)

    # calculate job-related values
    is_complete = None
    njobs = len(freqs)
    ijob = 0

    # check for existing file
    if os.path.isfile(file):
        if overwrite:  # if file exists, prompt for overwrite
            os.remove(file)  # remove existing file
            create_database(file, njobs)  # create database

        else: # continue from current progress
            is_complete, ijob = util.get_progress(file)
            if np.all(is_complete): return

    else:
        # Make directories if they do not exist
        file_dir = os.path.dirname(os.path.abspath(file))
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        # create database
        create_database(file, njobs)

    try:
        # start multiprocessing pool and run process
        write_lock = multiprocessing.Lock()
        pool = multiprocessing.Pool(threads, initializer=init_process, initargs=(write_lock,))
        jobs = util.create_jobs(file, (freqs, 1), mode='zip', is_complete=is_complete)
        result = pool.imap_unordered(run_process, jobs)

        for r in tqdm(result, desc='Benchmark', total=njobs, initial=ijob):
            pass

    except Exception as e:
        print(e)

    finally:

        pool.terminate()
        pool.close()


if __name__ == '__main__':
    main()