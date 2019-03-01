import numpy as np
import scipy as sp
import scipy.signal
from matplotlib import pyplot as plt

from cnld import abstract, simulation
from cnld.simulation import FixedStepSolver

eps = np.finfo(float).eps

# px = 0
# py = 0
# def load_func(x, y, prmin, prmax, pthmin, pthmax):
#     r = np.sqrt((x - px)**2 + (y - py)**2)
#     th = np.arctan2((y - py), (x - px))
#     th1 = th - 2 * eps
#     if th1 < -np.pi: 
#         th1 += 2 * np.pi
#     th2 = th + 2 * eps
#     if th2 > np.pi: 
#         th2 -= 2 * np.pi

#     if r - prmin >= -2 * eps:
#         if r - prmax <= 2 * eps:
#             if th1 - pthmin >= 0:
#                 if th1 - pthmax <= 0: 
#                     return 1, r, th1, th2
#             if th2 - pthmin >= 0:
#                 if th2 - pthmax <= 0:
#                     return 1, r, th1, th2
#     return 0, r, th1, th2



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

for i in range(4):
    plt.figure()
    plt.plot(v[mask[:,i], 0], v[mask[:,i], 1], '.')
    plt.gca().set_aspect('equal')

plt.show()

# fload = np.vectorize(load_func, excluded=[2,3,4,5])

# x = v[:,0]
# y = v[:,1]

# p = patches[0]
# mask, r, th1, th2 = fload(x, y, p.radius_min, p.radius_max, p.theta_min, p.theta_max)

# rmin = p.radius_min
# rmax = p.radius_max
# thmin = p.theta_min
# thmax = p.theta_max

# rmask = np.logical_and(r >= rmin, r <= rmax)
# th1mask = np.logical_and(th1 >= thmin, th1 <= thmax) 
# th2mask = np.logical_and(th2 >= thmin, th2 <= thmax)

v_t, v = simulation.linear_ramp(1e-6, 5e-9, tstop=t_stop)
v = 20 * v

solver = FixedStepSolver.from_array_and_db(array, db_file, v_t, v, t_start, t_stop, atol, maxiter=1)

fir = solver._fir
fir_t = solver._fir_t



fig, ax = plt.subplots(figsize=(11, 5))
plt.tight_layout()

ax.plot(fir_t, fir[0,0,:])
ax.plot(fir_t, fir[1,1,:])
ax.plot(fir_t, fir[2,2,:])
ax.plot(fir_t, fir[3,3,:])

ax.legend(range(4))

# plt.figure()
# plt.plot(fir_t, fir[4,:,:].T)
# plt.figure()
# plt.plot(fir_t, fir[5,:,:].T)
# plt.figure()
# plt.plot(fir_t, fir[6,:,:].T)
# plt.figure()
# plt.plot(fir_t, fir[7,:,:].T)

plt.show()



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