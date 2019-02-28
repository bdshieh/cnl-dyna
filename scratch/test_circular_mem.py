import numpy as np
import scipy as sp
import scipy.signal
from matplotlib import pyplot as plt

from cnld import abstract, simulation
from cnld.simulation import FixedStepSolver


array_file = 'circular_mem.json'
db_file = 'circular_mem.db'
t_start = 0
t_stop = 6e-6
atol = 1e-10

array = abstract.load(array_file)
patches = abstract.get_patches_from_array(array)


from cnld import mesh, fem

f = np.array(fem.f_from_abstract(array, refn=7).todense())
amesh = mesh.Mesh.from_abstract(array, refn=7)
v = amesh.vertices
mask = f > 0

for i in range(12):
    plt.figure()
    plt.plot(v[mask[:,i], 0], v[mask[:,i], 1], '.')
    plt.gca().set_aspect('equal')

plt.show()

# v_t, v = simulation.linear_ramp(1e-6, 5e-9, tstop=t_stop)
# v = 20 * v

# solver = FixedStepSolver.from_array_and_db(array, db_file, v_t, v, t_start, t_stop, atol, maxiter=1)

# fir = solver._fir
# fir_t = solver._fir_t






# fig, ax = plt.subplots(figsize=(11, 5))
# plt.tight_layout()

# ax.plot(fir_t, fir[0,8,:])
# ax.plot(fir_t, fir[8,0,:])

# fig.show()



# solver.solve()


# fig, ax = plt.subplots(figsize=(11, 5))
# tax = ax.twinx()
# plt.tight_layout()

# l1 = ax.plot(solver.time / 1e-6, solver.displacement[:, :] / 1e-9)
# # l1, = ax.plot(solver.time / 1e-6, np.mean(solver.displacement, axis=1) / 1e-9)
# l2, = tax.plot(v_t / 1e-6, v, '--',color='orange')

# # ax.legend(l1 + [l2,], ['1', '2', '3', 'voltage'])

# ax.set_xlim(0, 4)
# ax.set_ylim(-20, 0)
# tax.set_ylim(0, None)

# ax.set_xlabel(r'Time ($\mu s$)')
# ax.set_ylabel('Displacement (nm)')
# tax.set_ylabel('Voltage (V)')

# fig.show()