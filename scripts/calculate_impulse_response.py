
import numpy as np
import multiprocessing
import os
import sqlite3 as sql
from itertools import repeat
from contextlib import closing
from tqdm import tqdm
import traceback
import sys

from cmut_nonlinear_sim import bem, util

# register adapters for sqlite to convert numpy types
sql.register_adapter(np.float64, float)
sql.register_adapter(np.float32, float)
sql.register_adapter(np.int64, int)
sql.register_adapter(np.int32, int)

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
        query = 'INSERT INTO results VALUES (NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'
        con.executemany(query, row_data)


def create_frequencies_table(con, fs, ks):

    with con:

        # create table
        con.execute('CREATE TABLE frequencies (frequency float, wavenumber float, is_complete boolean)')

        # create indexes
        con.execute('CREATE UNIQUE INDEX frequency_index ON frequencies (frequency)')
        con.execute('CREATE UNIQUE INDEX wavenumber_index ON frequencies (wavenumber)')

        # insert values into table
        con.executemany('INSERT INTO frequencies VALUES (?, ?, ?)', zip(fs, ks, repeat(False)))


def create_nodes_table(con, nodes, membrane_ids, element_ids, channel_ids):

    x, y, z = nodes.T

    with con:

        # create table
        con.execute('CREATE TABLE nodes (x float, y float, z float, membrane_id, element_id, channel_id)')

        # create indexes
        con.execute('CREATE UNIQUE INDEX node_index ON nodes (x, y, z)')
        con.execute('CREATE INDEX membrane_id_index ON nodes (membrane_id)')
        con.execute('CREATE INDEX element_id_index ON nodes (element_id)')
        con.execute('CREATE INDEX channel_id_index ON nodes (channel_id)')

        # insert values into table
        query = 'INSERT INTO nodes VALUES (?, ?, ?, ?, ?, ?)'
        con.executemany(query, zip(x, y, z, membrane_ids, element_ids, channel_ids))


def create_displacements_table(con):

    with con:

        # create table
        query = '''
                CREATE TABLE displacements (
                id INTEGER PRIMARY KEY,
                frequency float,
                wavenumber float,
                x float,
                y float,
                z float,
                displacement_real float,
                displacement_imag float,
                FOREIGN KEY (frequency) REFERENCES frequencies (frequency),
                FOREIGN KEY (wavenumber) REFERENCES frequencies (wavenumber),
                FOREIGN KEY (x, y, z) REFERENCES nodes (x, y, z)
                )
                '''
        con.execute(query)

        # create indexes
        con.execute('CREATE INDEX query_index ON displacements (frequency, x, y, z)')


def update_displacements_table(con, f, k, nodes, displacements):

    x, y, z = nodes.T

    with con:
        query = '''
                INSERT INTO displacements (frequency, wavenumber, x, y, z, displacement_real, displacement_imag) 
                VALUES (?, ?, ?, ?, ?, ?, ?)
                '''
        con.executemany(query, zip(repeat(f), repeat(k), x, y, z, np.real(displacements.ravel()),
                                   np.imag(displacements.ravel())))



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


