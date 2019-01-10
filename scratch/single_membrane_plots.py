
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from cnld.mesh import square

freqs = []
x = []
nnodes = []

for refn in range(3,8):

    with np.load(f'febe_refn_{refn}.npz') as npf:
        freqs.append(npf['freqs'])
        x.append(npf['x']) 

    nnodes.append(len(square(1,1,refn=refn).vertices))

x_comsol_max = np.genfromtxt('comsol_max_disp.csv', delimiter=',', comments='%')
# x_comsol_mean = np.genfromtxt('comsol_mean_disp.csv', delimiter=',', comments='%')



fig, ax = plt.subplots(figsize=(7,4))
ax.plot(x_comsol_max[:,0] / 1e6, x_comsol_max[:,1],'o', markersize=3, fillstyle='none')
for i in range(len(x)):
    ax.plot(freqs[i] / 1e6, np.max(np.abs(x[i]), axis=0),'--', lw=1)
ax.set_xlabel('Frequency (MHz)')
ax.set_ylabel('Maximum displacement (m)')
labels = ['COMSOL'] + [f'{nnodes[i]} nodes' for i in range(len(x))]
ax.legend(labels)
plt.tight_layout()

fig, ax = plt.subplots(figsize=(7,4))
ax.plot(x_comsol_max[:,0] / 1e6, x_comsol_max[:,1],'o', markersize=3, fillstyle='none')
ax.plot(freqs[-1] / 1e6, np.max(np.abs(x[-1]), axis=0),'--', lw=1)
ax.set_xlabel('Frequency (MHz)')
ax.set_ylabel('Maximum displacement (m)')
ax.legend(['COMSOL', f'{nnodes[-1]} nodes'])
plt.tight_layout()

# fig, ax = plt.subplots(figsize=(7,4))
# ax.plot(x_comsol_mean[:,0] / 1e6, x_comsol_mean[:,1],'o', markersize=3, fillstyle='none')
# for i in range(len(x)):
#     ax.plot(freqs[i] / 1e6, np.mean(np.abs(x[i]), axis=0),'--', lw=1)
# ax.set_xlabel('Frequency (MHz)')
# ax.set_ylabel('Maximum displacement (m)')

plt.show()