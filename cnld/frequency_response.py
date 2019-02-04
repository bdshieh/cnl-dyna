'''
'''
import numpy as np
import sqlite3 as sql
import pandas as pd

from cnld import util

# register adapters for sqlite to convert numpy types
sql.register_adapter(np.float64, float)
sql.register_adapter(np.float32, float)
sql.register_adapter(np.int64, int)
sql.register_adapter(np.int32, int)
sql.register_adapter(np.uint64, int)
sql.register_adapter(np.uint32, int)


@util.open_db
def create_db(con, **kwargs):
    with con:
        create_displacements_table(con, **kwargs)


@util.open_db
def update_db(con, **kwargs):

    row_keys = ['frequency', 'wavenumber', 'source_patch', 'dest_patch', 'displacement_real', 
                'displacement_imag', 'time_solve']
    row_data = tuple([kwargs[k] for k in row_keys])

    with con:
        query = 'INSERT INTO displacements VALUES (?, ?, ?, ?, ?, ?, ?)'
        con.executemany(query, zip(*row_data))


@util.open_db
def create_displacements_table(con, **kwargs):

    with con:
        query = '''
                CREATE TABLE displacements (
                frequency float,
                wavenumber float,
                source_patch integer,
                dest_patch integer,
                displacement_real float,
                displacement_imag float,
                time_solve float
                )
                '''
        con.execute(query)

        # create indexes
        con.execute('CREATE INDEX frequency_index ON displacements (frequency)')
        con.execute('CREATE INDEX source_patch_index ON displacements (source_patch)')
        con.execute('CREATE INDEX dest_patch_index ON displacements (dest_patch)')


@util.open_db
def read_db(con):
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

    disp = np.array(table['displacement_real'] + 1j * table['displacement_imag']).reshape((nsource, ndest, nfreq))
    return freqs, disp