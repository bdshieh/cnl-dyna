''''''
__all__ = ['analyze_geometry']

import numpy as np
from matplotlib import pyplot as plt, patches
from cnld import fem
from cnld.api.grid import FemGrid


def analyze_geometry(geom,
                     refn=7,
                     nmode=6,
                     vstop=200,
                     plot_modes=True,
                     atol=1e-10,
                     maxiter=50):

    grid = FemGrid(geom, refn)
    eigf, eigv = fem.geom_eig(grid, geom)

    if geom.shape == 'square':
        xmax = 1.05 * geom.length_x / 2
        ymax = 1.05 * geom.length_y / 2

    elif geom.shape == 'circle':
        xmax = 1.05 * geom.radius
        ymax = 1.05 * geom.radius

    for i in range(nmode):
        f = eigf[i]
        v = eigv[:, i]

        xi, yi = np.meshgrid(np.linspace(-xmax, xmax, 41),
                             np.linspace(-ymax, ymax, 41))
        vi = grid.interpolate(xi, yi, v)
        vmax = np.max(np.abs(vi))

        if geom.shape == 'square':
            patch = patches.Rectangle(width=geom.length_x,
                                      height=geom.length_y,
                                      xy=(0, 0),
                                      ec='black',
                                      fill=False)
        elif geom.shape == 'circle':
            patch = patches.Circle(radius=geom.radius,
                                   xy=(0, 0),
                                   ec='black',
                                   fill=False)

        if plot_modes:
            fig, ax = plt.subplots()
            ax.add_patch(patch)
            ax.imshow(vi,
                      extent=(-xmax, xmax, -ymax, ymax),
                      vmax=vmax,
                      vmin=-vmax,
                      cmap='RdBu_r')
            ax.set_xlim(-xmax, xmax)
            ax.set_ylim(-ymax, ymax)
            ax.set_aspect('equal')
            ax.set_xlabel('x (m)')
            ax.set_ylabel('y (m)')
            ax.set_title(f'Mode {i}, f = {f:0.0f} Hz')

        print(f'Mode {i} freq: {f:0.0f} Hz')

    # determine collapse voltage
    for vdc in np.arange(vstop):

        x0, is_collapsed = fem.x_stat_vec_np(grid,
                                             geom,
                                             vdc,
                                             atol=atol,
                                             maxiter=maxiter)

        if is_collapsed:
            print(f'Snap-in voltage: {vdc} V')
            return

    raise RuntimeError('Snap-in voltage could not be determined')
