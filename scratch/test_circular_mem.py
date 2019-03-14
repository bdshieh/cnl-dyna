import numpy as np
import scipy as sp
import scipy.signal
from matplotlib import pyplot as plt
from scipy.integrate import dblquad

from cnld import abstract, simulation, mesh, fem
from cnld.simulation import FixedStepSolver

eps = np.finfo(float).eps


def circular_patch_f_vector(nodes, triangles, on_boundary, px, py, prmin, prmax, pthmin, pthmax):
    '''
    Load vector for a circular (polar) patch.
    '''
    def load_func(x, y):
        r = np.sqrt((x - px)**2 + (y - py)**2)
        th = np.arctan2((y - py), (x - px))
        # pertube theta by 2 * eps to account for potential round-off error
        th1 = th - 2 * eps
        if th1 < -np.pi: th1 += 2 * np.pi  # account for [-pi, pi] wrap-around
        th2 = th + 2 * eps
        if th2 > np.pi: th2 -= 2 * np.pi  # account for [-pi, pi] wrap-around
        if r - prmin >= -2 * eps:
            if r - prmax <= 2 * eps:
                # for theta, check both perturbed values
                if th1 - pthmin >= 0:
                    if th1 - pthmax <= 0:
                        return 1
                if th2 - pthmin >= 0:
                    if th2 - pthmax <= 0:
                        return 1
        return 0
    
    f = np.zeros(len(nodes))
    for tt in range(len(triangles)):
        tri = triangles[tt,:]
        xi, yi = nodes[tri[0],:2]
        xj, yj = nodes[tri[1],:2]
        xk, yk = nodes[tri[2],:2]

        # check if triangle vertices are inside or outside load
        loadi = load_func(xi, yi)
        loadj = load_func(xj, yj)
        loadk = load_func(xk, yk)
        # if load covers entire triangle
        # if all([loadi, loadj, loadk]):
        #     da = ((xj - xi) * (yk - yi) - (xk - xi) * (yj - yi))
        #     f[tri] += 1 / 6 * 2 * da
        # if load does not cover any part of triangle
        if not any([loadi, loadj, loadk]):
            continue
        # if load partially covers triangle
        else:
            def load_func_psi_eta(psi, eta):
                x = (xj - xi) * psi + (xk - xi) * eta + xi
                y = (yj - yi) * psi + (yk - yi) * eta + yi
                return load_func(x, y)

            frac, _ = dblquad(load_func_psi_eta, 0, 1, 0, lambda x: 1 - x, epsrel=1e-1, epsabs=1e-1)
            da = ((xj - xi) * (yk - yi) - (xk - xi) * (yj - yi))
            f[tri] += 1 / 6 * frac * da

    f[on_boundary] = 0

    return f


array_file = 'circular_mem_54.json'
db_file = 'circular_mem_54.db'
t_start = 0
t_stop = 6e-6
atol = 1e-10

array = abstract.load(array_file)
patches = abstract.get_patches_from_array(array)
mem = array.elements[0].membranes[0]

amesh = mesh.Mesh.from_abstract(array, refn=7)
nodes = amesh.vertices
f = np.zeros((len(amesh.vertices), len(patches)))
for i, pat in enumerate(patches):
    f[:,i] = circular_patch_f_vector(amesh.vertices, amesh.triangles, amesh.on_boundary,
        pat.position[0] - mem.position[0], pat.position[1] - mem.position[1], 
        pat.radius_min, pat.radius_max, pat.theta_min, pat.theta_max)

    f[amesh.on_boundary,i] = 0

mask = f > 0

# for i in range(4):
#     plt.figure()
#     plt.plot(nodes[mask[:,i], 0], nodes[mask[:,i], 1], '.')
#     plt.gca().set_aspect('equal')

# plt.figure()
# plt.plot(f[:,0:4], '.-')
# plt.figure()
# plt.plot(f[:,4:8], '.-')
# plt.figure()
# plt.plot(f[:,8:12], '.-')

# plt.show()

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



# fig, ax = plt.subplots(figsize=(11, 5))
# plt.tight_layout()
# plt.figure()
# for i in range(4):
#     plt.plot(fir_t, fir[i,i+4,:])
#     plt.plot(fir_t, fir[i+4,i,:])

# plt.legend(range(4))

# # plt.figure()
# # plt.plot(fir_t, fir[4,:,:].T)
# # plt.figure()
# # plt.plot(fir_t, fir[5,:,:].T)
# # plt.figure()
# # plt.plot(fir_t, fir[6,:,:].T)
# # plt.figure()
# # plt.plot(fir_t, fir[7,:,:].T)

# plt.show()



solver.solve()


fig, ax = plt.subplots(figsize=(11, 5))
tax = ax.twinx()
plt.tight_layout()

l1 = ax.plot(solver.time / 1e-6, solver.displacement[:, :] / 1e-9)
l1, = ax.plot(solver.time / 1e-6, np.mean(solver.displacement, axis=1) / 1e-9)
l2, = tax.plot(v_t / 1e-6, v, '--',color='orange')

# ax.legend(l1 + [l2,], ['1', '2', '3', 'voltage'])

ax.set_xlim(0, 4)
ax.set_ylim(-10, 0)
tax.set_ylim(0, None)

ax.set_xlabel(r'Time ($\mu s$)')
ax.set_ylabel('Displacement (nm)')
tax.set_ylabel('Voltage (V)')

fig.show()