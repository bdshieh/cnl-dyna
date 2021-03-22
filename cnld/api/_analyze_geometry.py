''''''
__all__ = [
    'analyze_geometry_modes', 'analyze_geometry_spectrum',
    'analyze_geometry_snapin'
]

import numpy as np
from matplotlib import pyplot as plt, patches
from tqdm import tqdm
from cnld import fem, bem
from cnld.api.grid import FemGrid


def analyze_geometry_modes(geom, refn=7, nmode=6, plot_modes=True):

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
            plt.show()

        print(f'Mode {i} freq: {f:0.0f} Hz')


def analyze_geometry_spectrum(geom,
                              refn=7,
                              rho=1000,
                              c=1500,
                              show_progress=True):

    grid = FemGrid(geom, refn)

    M = fem.m_mat_np(grid, geom)
    K = fem.k_mat_np(grid, geom)
    B = fem.b_eig_mat_np(grid, geom, M, K)
    P = fem.p_vec_np(grid, 1)

    freqs = np.linspace(0, 50e6, 1000)
    x = np.zeros(len(freqs))

    if show_progress:
        prog = tqdm
    else:
        prog = lambda x: x

    for i, f in prog(enumerate(freqs), total=len(freqs)):

        omg = 2 * np.pi * f
        k = omg / c
        Z = bem.z_mat_np_from_grid(grid, k)
        G = -(omg**2) * M - 1j * omg * B + K - omg**2 * 2 * rho * Z

        xf = np.linalg.solve(G, P)
        xf[grid.on_boundary] = 0
        x[i] = np.mean(np.abs(xf))

    fig, ax = plt.subplots()
    ax.plot(freqs / 1e6, 20 * np.log10(x / x.max()))
    ax.set_xlim(0, 50)
    ax.set_ylim(-100, 0)
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Magnitude (dB re max)')
    ax.set_title('Mean displacement spectrum')
    plt.show()


def analyze_geometry_snapin(geom,
                            refn=7,
                            vstop=200,
                            atol=1e-10,
                            maxiter=50,
                            show_progress=True):

    grid = FemGrid(geom, refn)

    if show_progress:
        prog = tqdm
    else:
        prog = lambda x: x

    # determine collapse voltage
    for vdc in prog(np.arange(vstop)):

        x0, is_collapsed = fem.x_stat_vec_np(grid,
                                             geom,
                                             vdc,
                                             atol=atol,
                                             maxiter=maxiter)

        if is_collapsed:
            print(f'Snap-in voltage: {vdc} V')
            return

    raise RuntimeError('Snap-in voltage could not be determined')