
from cmut_nonlinear_sim.core.libh2_classes import *
from cmut_nonlinear_sim.core.libh2_functions import *

import numpy as np


def new_square_macrosurface(ax, bx):

    ms = Macrosurface(4, 5, 2)

    #  vertices 
    #  bottom left 
    ms.x[0][0] = -ax / 2
    ms.x[0][1] = -bx / 2
    ms.x[0][2] = 0.0
    #  bottom right 
    ms.x[1][0] = ax / 2
    ms.x[1][1] = -bx / 2
    ms.x[1][2] = 0.0
    #  top right 
    ms.x[2][0] = ax / 2
    ms.x[2][1] = bx / 2
    ms.x[2][2] = 0.0
    #  top left 
    ms.x[3][0] = -ax / 2
    ms.x[3][1] = bx / 2
    ms.x[3][2] = 0.0

    #  vertex edges 
    #  bottom 
    ms.e[0][0] = 0
    ms.e[0][1] = 1
    #  right 
    ms.e[1][0] = 1
    ms.e[1][1] = 2
    #  top 
    ms.e[2][0] = 2
    ms.e[2][1] = 3
    #  left 
    ms.e[3][0] = 3
    ms.e[3][1] = 0
    #  diagonal 
    ms.e[4][0] = 1
    ms.e[4][1] = 3

    #  triangles and triangle edges 
    #  bottom left 
    ms.t[0][0] = 0
    ms.t[0][1] = 1
    ms.t[0][2] = 3
    ms.s[0][0] = 4
    ms.s[0][1] = 3
    ms.s[0][2] = 0
    #  top right 
    ms.t[1][0] = 1
    ms.t[1][1] = 2
    ms.t[1][2] = 3
    ms.s[1][0] = 2
    ms.s[1][1] = 4
    ms.s[1][2] = 1

    return ms




if __name__ == '__main__':

	m = 4
	clf = 2 * m * m * m # /* Minimal leaf size for cluster tree construction. */
	eta = 1.4 # /* Parameter 'eta' within the admissibilty condition. */
	k = 2 * np.pi * 1e6 / 1540.0 # /* wavenumber */
	accur = 1e-12 # /* accuracy of rk as a fraction of the norm */
	ax = 100e-6
	bx = 100e-6
	refn = 16
	q_reg = 2
	q_sing = q_reg + 2
	basis = basistype.CONSTANT
	alpha = 0.0

	init_h2lib()

	# /****************************************************
	# * Create geometry
	# ****************************************************/
	

	ms = new_square_macrosurface(ax, bx)
	surf = build_from_macrosurface3d_surface3d(ms, refn)
	# print("Mesh:\n")
	# print("  %u vertices\n", surf.vertices)
	# print("  %u edges\n", surf.edges)
	# print("  %u triangles\n", surf.triangles)

	# /****************************************************
	# * Set up H-matrix
	# ****************************************************/

	bem_slp = new_slp_helmholtz_bem3d(k, surf, q_reg, q_sing, basis, basis)

	bem_dlp = new_dlp_helmholtz_bem3d(k, surf, q_reg, q_sing, basis, basis, alpha)

	# /* create cluster tree. */
	root = build_bem3d_cluster(bem_slp, clf, basis)

	# /* create block tree. */
	broot = build_nonstrict_block(root, root, eta, '2')

	# # /* Set up interpolation approximation scheme for H-matrix V. */
	# # //setup_hmatrix_aprx_inter_row_bem3d(bem_slp, root, root, broot, m)
	setup_hmatrix_aprx_paca_bem3d(bem_slp, root, root, broot, accur)
	# # //setup_hmatrix_aprx_hca_bem3d(bem_slp, root, root, broot, m, accur)

	# # /****************************************************
	# # * Assemble H-matrix
	# # ****************************************************/

	# # /* create h-matrix structure from block tree. */
	# Z = build_from_block_hmatrix(broot, m * m * m)

	# start_stopwatch(sw)
	# # /* assemble near- and farfield entries of v. */
	# assemble_bem3d_hmatrix(bem_slp, broot, Z)
	# t = stop_stopwatch(sw)
	# # /* get the total memory footprint for v. */
	# size = getsize_hmatrix(Z) / 1024.0 / 1024.0

	# print("H-matrix:\n")
	# print("  Assembly %.2f s\n", t)
	# print("  Storage %.3f mb\n", size)

	# # /****************************************************
	# # * Set up RHS
	# # ****************************************************/

	# x = new_avector(gr.vertices)
	# random_avector(x)

	# b = new_avector(gr.vertices)
	# clear_avector(b)
	# addevalsymm_hmatrix_avector(alpha, Z, x, b)


	# # /****************************************************
	# # * Set up solver and decomposition parameters
	# # ****************************************************/

	# tm = new_releucl_truncmode()
	# eps = 1e-12
	# maxiter = 1000

	# # /****************************************************
	# # * Solve with H-matrix
	# # ****************************************************/

	# x_cg = new_zero_avector(gr.vertices)

	# start_stopwatch(sw)
	# solve_cg_hmatrix_avector(Z, b, x_cg, eps, maxiter)
	# t = stop_stopwatch(sw)

	# # /* Calculate RMSE referenced to H-matrix solution */
	# add_avector(-alpha, x, x_cg)
	# rmse = norm2_avector(x_cg) / norm2_avector(x)

	# print("  Solve: %.2f s\n", t)
	# print("  Error: %.2f \n", rmse)
	# fflush(stdout)

	# # /****************************************************
	# # * LU factorize H-matrix
	# # ****************************************************/

	# Z_lu = build_from_block_hmatrix(broot, m * m * m)
	# x_lu = new_avector(gr.vertices)

	# copy_hmatrix(Z, Z_lu)
    # copy_avector(b, x_lu)

    # start_stopwatch(sw)
	# lrdecomp_hmatrix(Z_lu, tm, eps)
	# t = stop_stopwatch(sw)
	# size = getsize_hmatrix(Z_lu) / 1024.0 / 1024.0

	# print("LU:\n")
	# print("  Factorization: %.2f s\n", t)
	# print("  Storage: %.3f mb\n", size)

	# start_stopwatch(sw)
	# lrsolve_hmatrix_avector(false, Z_lu, x_lu)
	# t = stop_stopwatch(sw)

	# # /* Calculate RMSE referenced to H-matrix solution */
	# add_avector(-alpha, x, x_lu)
	# rmse = norm2_avector(x_lu) / norm2_avector(x)

	# print("  Solve: %.2f s\n", t)
	# print("  Error: %.2f \n", rmse)
	# fflush(stdout)

	# # /****************************************************
	# # * Cholesky factorize H-matrix
	# # ****************************************************/

	# Z_chol = build_from_block_hmatrix(broot, m * m * m)
	# x_chol = new_avector(gr.vertices)

	# copy_hmatrix(Z, Z_chol)
	# copy_avector(b, x_chol)

	# start_stopwatch(sw)
	# choldecomp_hmatrix(Z_chol, tm, eps)
	# t = stop_stopwatch(sw)
	# size = getsize_hmatrix(Z_chol) / 1024.0 / 1024.0

	# print("Cholesky:\n")
	# print("  Factorization: %.2f s\n", t)
	# print("  Storage: %.3f mb\n", size)

	# start_stopwatch(sw)
	# cholsolve_hmatrix_avector(Z_chol, x_chol)
	# t = stop_stopwatch(sw)

	# # /* Calculate RMSE referenced to H-matrix solution */
	# add_avector(-alpha, x, x_chol)
	# rmse = norm2_avector(x_chol) / norm2_avector(x)

	# uninit_h2lib()