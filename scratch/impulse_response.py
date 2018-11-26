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
        util.create_progress_table(con, **kwargs)
        create_frequencies_table(con, **kwargs)
        create_mesh_tables(con, **kwargs)
        create_displacements_table(con, **kwargs)


@util.open_db
def update_database(con, **kwargs):

    row_keys = ['frequency', 'wavenumber', 'x', 'y', 'z', 'membrane_id', 'element_id', 'displacement_real', 'displacement_imag']
    row_data = [kwargs[k] for k in row_keys]

    with con:
        append_displacements_table(con, row_data)


@util.open_db
def create_mesh_tables(con, **kwargs):

    x, y, z = kwargs['vertices'].T
    edges = kwargs['edges']
    triangles = kwargs['triangles']
    triangle_edges = kwargs['triangle_edges']
    membrane_ids = kwargs['membrane_id']
    element_ids = kwargs['element_id']

    with con:

        # create table
        query = '''
                CREATE TABLE vertices (
                id INTEGER PRIMARY KEY, 
                x float, 
                y float, 
                z float, 
                membrane_id integer, 
                element_id integer
                )
                '''
        con.execute(query)

        # create indexes
        con.execute('CREATE INDEX vertex_index ON vertices (x, y, z)')
        con.execute('CREATE INDEX membrane_id_index ON vertices (membrane_id)')
        con.execute('CREATE INDEX element_id_index ON vertices (element_id)')

        # insert values into table
        query = 'INSERT INTO vertices VALUES (NULL, ?, ?, ?, ?, ?)'
        con.executemany(query, zip(x, y, z, membrane_ids, element_ids))

        # create table
        query = '''
                CREATE TABLE edges (
                id INTEGER PRIMARY KEY, 
                v0 integer, 
                v1 integer, 
                FOREIGN KEY (v0, v1) REFERENCES vertices (id, id))
                '''
        con.execute(query)

        # insert values into table
        query = 'INSERT INTO edges VALUES (NULL, ?, ?)'
        con.executemany(query, edges)

        # create table
        query = '''
                CREATE TABLE triangles (
                id INTEGER PRIMARY KEY, 
                v0 integer, 
                v1 integer, 
                v2 integer,
                FOREIGN KEY (v0, v1, v2) REFERENCES vertices (id, id, id))
                '''
        con.execute(query)

        # insert values into table
        query = 'INSERT INTO triangles VALUES (NULL, ?, ?, ?)'
        con.executemany(query, triangles)

        # create table
        query = '''
                CREATE TABLE triangle_edges (
                id INTEGER PRIMARY KEY, 
                e0 integer, 
                e1 integer, 
                e2 integer,
                FOREIGN KEY (e0, e1, e2) REFERENCES edges (id, id, id))
                '''
        con.execute(query)

        # insert values into table
        query = 'INSERT INTO triangle_edges VALUES (NULL, ?, ?, ?)'
        con.executemany(query, triangle_edges)


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
                FOREIGN KEY (frequency, wavenumber) REFERENCES frequencies (frequency, wavenumber),
                FOREIGN KEY (x, y, z) REFERENCES vertices (x, y, z)
                )
                '''
        con.execute(query)


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
    data['vertices'] = mesh.vertices
    data['edges'] = mesh.edges
    data['triangles'] = mesh.triangles
    data['triangle_edges'] = mesh.triangle_edges
    data['frequencies'] = freqs
    data['wavenumbers'] = wavenums
    data['njobs'] = len(freqs)
    data['membrane_id'] = np.zeros(len(mesh.vertices))
    data['element_id'] = np.zeros(len(mesh.vertices))

    create_database('test.db', **data)


