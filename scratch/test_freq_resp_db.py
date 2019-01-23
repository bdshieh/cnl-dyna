
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
# from cnld.impulse_response import read_freq_resp_db
from cnld import util


@util.open_db
def read_freq_resp_db(con):
    with con:
        query = '''
                SELECT source_patch, dest_patch, frequency, displacement_real, displacement_imag FROM displacements
                ORDER BY source_patch, dest_patch, frequency
                '''
        table = pd.read_sql(query, con)
        
    source_patch = np.unique(table['source_patch'].values)
    dest_patch = np.unique(table['dest_patch'].values)
    freqs = np.unique(table['frequency'].values)
    nsource = len(source_patch)
    ndest = len(dest_patch)
    nfreq = len(freqs)
    assert nsource == ndest

    disp = np.array(table['displacement_real'] + 1j * table['displacement_imag']).reshape((nsource, ndest, nfreq))
    return disp, freqs


if __name__ == '__main__':

    db, freqs = read_freq_resp_db('freq-resp.db')

    plt.imshow(20 * np.log10(np.abs(db[...,20])), cmap='RdBu_r')
    plt.show()

    parea = (40e-6 / 3)**2
    dbs = db.sum(0).sum(0) / 9
    plt.plot(freqs / 1e6, np.abs(dbs))
    plt.show()

    # print(np.max(np.abs(db)))

    # a = db[:,4,-1]
    # b = a.reshape((3,3,5,5), order='F')
    # bmax = np.abs(b).max()
    # blog = 20 * np.log10(np.abs(b) / bmax)
    # blog[blog < -100] = -100

    # fig, axs = plt.subplots(5, 5, figsize=(9,9))
    # for i in range(5):
    #     for j in range(5):
    #         axs[i, j].imshow(blog[:,:,i,j], vmax=0, vmin=-60, cmap='RdBu_r')
    #         axs[i, j].set_aspect('equal')
    #         axs[i, j].set_axis_off()
    # plt.tight_layout()
    # fig.show()