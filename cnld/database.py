'''
'''
import numpy as np
import sqlite3 as sql
import pandas as pd
from contextlib import closing

from cnld import util

# register adapters for sqlite to convert numpy types
sql.register_adapter(np.float64, float)
sql.register_adapter(np.float32, float)
sql.register_adapter(np.int64, int)
sql.register_adapter(np.int32, int)
sql.register_adapter(np.uint64, int)
sql.register_adapter(np.uint32, int)


''' TABLE DEFINITIONS '''

NODE = '''
CREATE TABLE node (
node_id integer primary key,
x float,
y float,
z float
)
'''

PATCH_TO_PATCH_FREQ_RESP = '''
CREATE TABLE patch_to_patch_freq_resp (
source_patch integer,
dest_patch integer,
frequency float,
wavenumber float,
displacement_real float,
displacement_imag float,
time_solve float
)
'''
PATCH_TO_PATCH_FREQ_RESP_INDEX = '''
CREATE INDEX patch_to_patch_freq_resp_index ON patch_to_patch_freq_resp (
source_patch, dest_patch, frequency
)
'''

PATCH_TO_NODE_FREQ_RESP = '''
CREATE TABLE patch_to_node_freq_resp (
source_patch integer,
node_id integer,
x float,
y float,
z float,
frequency float,
wavenumber float,
displacement_real float,
displacement_imag float,
FOREIGN KEY (node_id, x, y, z) REFERENCES node (node_id, x, y, z)
)
'''
PATCH_TO_NODE_FREQ_RESP_INDEX = '''
CREATE INDEX patch_to_node_freq_resp_index ON patch_to_node_freq_resp (
source_patch, node_id, frequency
)
'''

PATCH_TO_PATCH_IMP_RESP = '''
CREATE TABLE patch_to_patch_imp_resp (
source_patch integer,
dest_patch integer,
time float,
displacement float
)
'''
PATCH_TO_PATCH_IMP_RESP_INDEX = '''
CREATE INDEX patch_to_patch_imp_resp_index ON patch_to_patch_imp_resp (
source_patch, dest_patch, time
)
'''

# PROGRESS =
# '''
# CREATE TABLE progress (
# job_id integer primary key, 
# is_complete boolean
# )
# '''


''' CREATING DATABASE '''

@util.open_db
def create_table(con, table_defn, index_defn=None):

    with con:
        con.execute(table_defn)
        if index_defn is not None:
            con.execute(index_defn)


@util.open_db
def create_db(con):

    # create_table(con, PROGRESS)
    create_table(con, NODE)
    create_table(con, PATCH_TO_PATCH_FREQ_RESP, PATCH_TO_PATCH_FREQ_RESP_INDEX)
    create_table(con, PATCH_TO_NODE_FREQ_RESP, PATCH_TO_NODE_FREQ_RESP_INDEX)
    create_table(con, PATCH_TO_PATCH_IMP_RESP, PATCH_TO_PATCH_IMP_RESP_INDEX)


''' UPDATING DATABASE '''

@util.open_db
def append_node(con, **kwargs):

    row_keys = ['node_id', 'x', 'y', 'z']
    row_data = tuple([kwargs[k] for k in row_keys])

    with con:
        query = f'INSERT INTO node VALUES (?,?,?,?)'
        con.executemany(query, zip(*row_data))


@util.open_db
def append_patch_to_patch_freq_resp(con, **kwargs):

    row_keys = ['source_patch', 'dest_patch', 'frequency', 'wavenumber', 'displacement_real', 
        'displacement_imag', 'time_solve']
    row_data = tuple([kwargs[k] for k in row_keys])

    with con:
        query = f'INSERT INTO patch_to_patch_freq_resp VALUES (?,?,?,?,?,?,?)'
        con.executemany(query, zip(*row_data))


@util.open_db
def append_patch_to_node_freq_resp(con, **kwargs):

    row_keys = ['source_patch', 'node_id', 'x', 'y', 'z', 'frequency', 'wavenumber',  
        'displacement_real', 'displacement_imag']
    row_data = tuple([kwargs[k] for k in row_keys])

    with con:
        query = 'INSERT INTO patch_to_node_freq_resp VALUES (?,?,?,?,?,?,?,?,?)'
        con.executemany(query, zip(*row_data))


@util.open_db
def append_patch_to_patch_imp_resp(con, **kwargs):
    ''''''
    row_keys = ['source_patch', 'dest_patch', 'time', 'displacement']
    row_data = tuple([kwargs[k] for k in row_keys])

    with con:
        query = 'INSERT INTO patch_to_patch_imp_resp VALUES (?, ?, ?, ?)'
        con.executemany(query, zip(*row_data))


# @util.open_db
# def update_progress(con, job_id):

#     with con:
#         con.execute('UPDATE progress SET is_complete=1 WHERE job_id=?', [job_id,])


''' READ FROM DATABASE '''

# @util.open_db
# def get_progress(con):

#     table = pd.read_sql('SELECT is_complete FROM progress ORDER BY job_id', con)

#     is_complete = np.array(table).squeeze()
#     ijob = sum(is_complete) + 1

#     return is_complete, ijob


@util.open_db
def read_node(con):
    with con:
        query = '''
                SELECT node_id, x, y, z
                FROM node
                ORDER BY node_id
                '''
        table = pd.read_sql(query, con)
        
    return np.array([table['x'], table['y'], table['z']])


@util.open_db
def read_patch_to_patch_freq_resp(con):
    with con:
        query = '''
                SELECT source_patch, dest_patch, frequency, displacement_real, displacement_imag 
                FROM patch_to_patch_freq_resp
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


@util.open_db
def read_patch_to_node_freq_resp(con):
    with con:
        query = '''
                SELECT source_patch, node_id, x, y, z, frequency, displacement_real, displacement_imag 
                FROM patch_to_node_freq_resp
                ORDER BY source_patch, node_id, frequency
                '''
        table = pd.read_sql(query, con)
        
    source_patch_ids = np.unique(table['source_patch'].values)
    freqs = np.unique(table['frequency'].values)
    nodes = np.array(table[table['source_patch'] == 0][table['frequency'] == freqs[0]][['x', 'y', 'z']])
    
    nsource = len(source_patch_ids)
    nnodes = len(nodes)
    nfreq = len(freqs)

    disp = np.array(table['displacement_real'] + 1j * table['displacement_imag']).reshape((nsource, nnodes, nfreq))
    return freqs, disp, nodes


@util.open_db
def read_patch_to_patch_imp_resp(con):
    with con:
        query = '''
                SELECT source_patch, dest_patch, time, displacement 
                FROM patch_to_patch_imp_resp
                ORDER BY source_patch, dest_patch, time
                '''
        table = pd.read_sql(query, con)
        
    source_patches = np.unique(table['source_patch'].values)
    dest_patches = np.unique(table['dest_patch'].values)
    times = np.unique(table['time'].values)

    nsource = len(source_patches)
    ndest = len(dest_patches)
    ntime = len(times)

    disp = np.array(table['displacement']).reshape((nsource, ndest, ntime))
    return times, disp