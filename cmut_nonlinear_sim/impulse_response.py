## cmut_nonlinear_sim / impulse_response.py ##

import numpy as np
import os
import sqlite3 as sql

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

    row_keys = ['frequency', 'wavenumber', 'x', 'y', 'z', 'membrane_id', 'element_id', 'displacement_real', 'displacement_imag']
    row_data = [kwargs[k] for k in row_keys]

    with con:
        append_displacements_table(con, row_data)


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
        query = '''
                CREATE TABLE displacements (
                id INTEGER PRIMARY KEY,
                frequency float,
                wavenumber float,
                x float,
                y float,
                z float,
                membrane_id integer,
                element_id integer,
                displacement_real float,
                displacement_imag float,
                FOREIGN KEY (frequency, wavenumber) REFERENCES frequencies (frequency, wavenumber)
                )
                '''
        con.execute(query)

        # create indexes
        con.execute('CREATE INDEX vertex_index ON displacements (x, y, z)')
        con.execute('CREATE INDEX membrane_id_index ON displacements (membrane_id)')
        con.execute('CREATE INDEX element_id_index ON displacements (element_id)')


@util.open_db
def append_displacements_table(con, row_data):

    with con:
        query = 'INSERT INTO displacements VALUES (NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?)'
        con.executemany(query, row_data)



if __name__ == '__main__':
    
    from . mesh import fast_matrix_array

    mesh = fast_matrix_array(5, 5, 60e-6, 60e-6, refn=3, lengthx=40e-6, lengthy=40e-6)
    freqs = np.arange(50e3, 30e6 + 50e3, 50e3)
    wavenums = 2 * np.pi * freqs / 1500

    data = {}
    data['frequencies'] = freqs
    data['wavenumbers'] = wavenums

    create_database('test.db', **data)


