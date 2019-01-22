## cmut_nonlinear_sim / impulse_response.py ##

import numpy as np
import os
import sqlite3 as sql
import pandas as pd

from . import util

# register adapters for sqlite to convert numpy types
sql.register_adapter(np.float64, float)
sql.register_adapter(np.float32, float)
sql.register_adapter(np.int64, int)
sql.register_adapter(np.int32, int)
sql.register_adapter(np.uint64, int)
sql.register_adapter(np.uint32, int)


@util.open_db
def create_database(con, **kwargs):
    with con:
        create_frequencies_table(con, **kwargs)
        create_displacements_table(con, **kwargs)


@util.open_db
def update_database(con, **kwargs):

#     row_keys = ['frequency', 'wavenumber', 'source_patch_id', 'dest_patch_id', 'source_membrane_id',
#                 'dest_membrane_id', 'source_element_id', 'dest_element_id', 'displacement_real', 
#                 'displacement_imag']
    row_keys = ['frequency', 'wavenumber', 'source_patch', 'dest_patch', 'displacement_real', 
                'displacement_imag', 'time_solve', 'iterations']
    row_data = tuple([kwargs[k] for k in row_keys])

    with con:
        query = 'INSERT INTO displacements VALUES (NULL, ?, ?, ?, ?, ?, ?, ?, ?)'
        con.executemany(query, zip(*row_data))


@util.open_db
def create_frequencies_table(con, **kwargs):

    fs = kwargs['frequencies']
    ks = kwargs['wavenumbers']

    with con:
        # create table
        con.execute('CREATE TABLE frequencies (id INTEGER PRIMARY KEY, frequency float, wavenumber float)')
        # create indexes
        con.execute('CREATE UNIQUE INDEX frequency_index ON frequencies (frequency)')
        con.execute('CREATE UNIQUE INDEX wavenumber_index ON frequencies (wavenumber)')
        # insert values into table
        con.executemany('INSERT INTO frequencies VALUES (NULL, ?, ?)', zip(fs, ks))


@util.open_db
def create_displacements_table(con, **kwargs):

    with con:
        # create table
        # query = '''
        #         CREATE TABLE displacements (
        #         id INTEGER PRIMARY KEY,
        #         frequency float,
        #         wavenumber float,
        #         source_patch_id integer,
        #         dest_patch_id integer,
        #         source_membrane_id integer,
        #         dest_membrane_id integer,
        #         source_element_id integer,
        #         dest_element_id integer,
        #         displacement_real float,
        #         displacement_imag float,
        #         FOREIGN KEY (frequency, wavenumber) REFERENCES frequencies (frequency, wavenumber)
        #         )
        #         '''
        query = '''
                CREATE TABLE displacements (
                id INTEGER PRIMARY KEY,
                frequency float,
                wavenumber float,
                source_patch integer,
                dest_patch integer,
                displacement_real float,
                displacement_imag float,
                time_solve float,
                iterations integer,
                FOREIGN KEY (frequency, wavenumber) REFERENCES frequencies (frequency, wavenumber)
                )
                '''
        con.execute(query)

        # create indexes
        con.execute('CREATE INDEX displacements_index ON displacements (frequency)')
        con.execute('CREATE INDEX source_patch_index ON displacements (source_patch)')
        con.execute('CREATE INDEX dest_patch_index ON displacements (dest_patch)')
        # con.execute('CREATE INDEX source_membrane_id_index ON displacements (source_membrane_id)')
        # con.execute('CREATE INDEX dest_membrane_id_index ON displacements (dest_membrane_id)')
        # con.execute('CREATE INDEX source_element_id_index ON displacements (source_element_id)')
        # con.execute('CREATE INDEX dest_element_id_index ON displacements (dest_element_id)')


@util.open_db
def read_freq_resp_db(con):
    with con:
        query = '''
                SELECT source_patch, dest_patch, frequency, displacement_real, displacement_imag FROM displacements
                ORDER BY source_patch, dest_patch, frequency
                '''
        table = pd.read_sql(query, con)
        
    source_patch_ids = np.unique(table['source_patch'].values)
    dest_patch_ids = np.unique(table['dest_patch'].values)
    freqs = np.unique(table['frequency'].values)
    nsource = len(source_patch_ids)
    ndest = len(dest_patch_ids)
    nfreq = len(freqs)
    assert nsource == ndest

    disp = np.array(table['displacement_real'] + 1j * table['displacement_imag']).reshape((nsource, ndest, nfreq), 
        order='F')
    return disp, freqs


if __name__ == '__main__':
    
    from . mesh import fast_matrix_array

    mesh = fast_matrix_array(5, 5, 60e-6, 60e-6, refn=3, xl=40e-6, yl=40e-6)
    freqs = np.arange(50e3, 30e6 + 50e3, 50e3)
    wavenums = 2 * np.pi * freqs / 1500

    data = {}
    data['frequencies'] = freqs
    data['wavenumbers'] = wavenums

    create_database('test.db', **data)


