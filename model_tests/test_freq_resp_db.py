
import numpy as np
from matplotlib import pyplot as plt
from cnld.frequency_response import read_db
from cnld import impulse_response, util





if __name__ == '__main__':

    freqs, db = read_db('freq_resp.db')

    t, fir = impulse_response.fft_to_fir(freqs, db)

    plt.plot(t, fir[4,4,:])
    plt.show()
    
    # plt.figure()
    # plt.imshow(20 * np.log10(np.abs(db[...,15])), cmap='RdBu_r')
    
    # dbs = db.sum(0).reshape((3,3,2,-1), order='F')

    # plt.figure()
    # plt.imshow(np.abs(dbs[:,:,0,15]), cmap='RdBu_r')
    # plt.figure()
    # plt.imshow(np.abs(dbs[:,:,1,15]), cmap='RdBu_r')
    # plt.figure()
    # plt.plot(freqs / 1e6, np.abs(db.sum(0).sum(0) / 9))
    
    # plt.show()

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