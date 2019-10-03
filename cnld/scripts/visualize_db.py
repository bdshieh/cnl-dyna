'''
Generates figures to visualize impulse and frequency responses from a database.
'''
import numpy as np
from cnld import database
from matplotlib import pyplot as plt

max_number_of_patches = 10

if __name__ == '__main__':

    file = sys.argv[1]

    t, ppir = database.read_patch_to_patch_imp_resp(file)
    f, ppfr = database.read_patch_to_patch_freq_resp(file)
    npatch = ppir.shape[0]
    nplot = min(npatch, max_number_of_patches)

    metadata = database.read_metadata(file)

    print(metadata.iloc[0])

    for i in range(nplot):

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4))
        ax1.plot(t / 1e-6, ppir[i, :nplot, :].T, lw=1)
        ax1.set_xlim(t.min() / 1e-6, t.max() / 1e-6 / 2)
        ax1.set_xlabel('Time ($\mu s$)')
        ax1.set_ylabel('Displacement / Pressure (m / Pa)')
        ax1.legend(np.arange(0, nplot), ncol=2)
        ax1.set_title('Patch ' + str(i) + ' Impulse Response')

        ax2.plot(f / 1e6, 20 * np.log10(np.abs(ppfr[i, :nplot, :].T)), lw=1)
        ax2.set_xlim(f.min() / 1e-6, f.max() / 1e6)
        ax2.set_xlabel('Frequency ($MHz$)')
        ax2.set_ylabel('Displacement / Pressure (m / Pa)')
        ax2.legend(np.arange(0, nplot), ncol=2)
        ax2.set_title('Patch ' + str(i) + ' Frequency Response')

        fig.show()
