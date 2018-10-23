#include <stdio.h>

#include "basic.h"
#include "macrosurface3d.h"
#include "helmholtzbem3d.h"
#include "krylovsolvers.h"

static void cube_parametrization(uint i, real xr1, real xr2, void * data, real xt[3]);
pmacrosurface3d new_square_macrosurface3d(real ax, real bx);


static void cube_parametrization(uint i, real xr1, real xr2, void *data, real xt[3])
{
	pcmacrosurface3d mg = (pcmacrosurface3d)data;
	const     real(*x)[3] = (const real(*)[3]) mg->x;
	const     uint(*t)[3] = (const uint(*)[3]) mg->t;

	assert(i < mg->triangles);
	assert(t[i][0] < mg->vertices);
	assert(t[i][1] < mg->vertices);
	assert(t[i][2] < mg->vertices);

	xt[0] = (x[t[i][0]][0] * (1.0 - xr1 - xr2) + x[t[i][1]][0] * xr1
		+ x[t[i][2]][0] * xr2);
	xt[1] = (x[t[i][0]][1] * (1.0 - xr1 - xr2) + x[t[i][1]][1] * xr1
		+ x[t[i][2]][1] * xr2);
	xt[2] = (x[t[i][0]][2] * (1.0 - xr1 - xr2) + x[t[i][1]][2] * xr1
		+ x[t[i][2]][2] * xr2);
}


pmacrosurface3d new_square_macrosurface3d(real ax, real bx) 
{
	pmacrosurface3d mg = new_macrosurface3d(4, 5, 2);

	/* vertices */
	/* bottom left */
	mg->x[0][0] = -ax / 2;
	mg->x[0][1] = -bx / 2;
	mg->x[0][2] = 0.0;
	/* bottom right */
	mg->x[1][0] = ax / 2;
	mg->x[1][1] = -bx / 2;
	mg->x[1][2] = 0.0;
	/* top right */
	mg->x[2][0] = ax / 2;
	mg->x[2][1] = bx / 2;
	mg->x[2][2] = 0.0;
	/* top left */
	mg->x[3][0] = -ax / 2;
	mg->x[3][1] = bx / 2;
	mg->x[3][2] = 0.0;

	/* vertex edges */
	/* bottom */
	mg->e[0][0] = 0;
	mg->e[0][1] = 1;
	/* right */
	mg->e[1][0] = 1;
	mg->e[1][1] = 2;
	/* top */
	mg->e[2][0] = 2;
	mg->e[2][1] = 3;
	/* left */
	mg->e[3][0] = 3;
	mg->e[3][1] = 0;
	/* diagonal */
	mg->e[4][0] = 1;
	mg->e[4][1] = 3;

	/* triangles and triangle edges */
	/* bottom left */
	mg->t[0][0] = 0;
	mg->t[0][1] = 1;
	mg->t[0][2] = 3;
	mg->s[0][0] = 4;
	mg->s[0][1] = 3;
	mg->s[0][2] = 0;
	/* top right */
	mg->t[1][0] = 1;
	mg->t[1][1] = 2;
	mg->t[1][2] = 3;
	mg->s[1][0] = 2;
	mg->s[1][1] = 4;
	mg->s[1][2] = 1;

	mg->phi = cube_parametrization;
	mg->phidata = mg;

	return mg;
}


int main(int argc, char **argv)
{
	real ax, bx;
	uint refn, q_reg, q_sing;
	basisfunctionbem3d basis;
	uint m, clf;
	real k;
	real eta, accur;
	pstopwatch sw;
	pmacrosurface3d mg;
	psurface3d gr;
	pbem3d bem_slp;
	real t, size;
	pcluster  root;
	pblock broot;
	phmatrix Z, Z_lu, Z_chol;
	pavector x, x_cg, x_lu, x_chol, b;
	ptruncmode tm;
	real eps, maxiter, rmse;
	field alpha = 1.0;

	/* Init the H2Lib, should be called before any other function. */
	init_h2lib(&argc, &argv);

	/****************************************************
	* Set up basic parameters
	****************************************************/

	m = 4;
	clf = 2 * m * m * m; /* Minimal leaf size for cluster tree construction. */
	eta = 1.4; /* Parameter 'eta' within the admissibilty condition. */
	k = 2 * M_PI * 1e6 / 1540.0; /* wavenumber */
	accur = 1e-12; /* accuracy of rk as a fraction of the norm */
	ax = 100e-6;
	bx = 100e-6;
	refn = 32;
	q_reg = 2;
	q_sing = q_reg + 2;
	basis = BASIS_LINEAR_BEM3D;

	sw = new_stopwatch(); /* Stopwatch for measuring the time. */

	/****************************************************
	* Create geometry
	****************************************************/

	mg = new_square_macrosurface3d(ax, bx);
	gr = build_from_macrosurface3d_surface3d(mg, refn);
	printf("Mesh:\n");
	printf("  %u vertices\n", gr->vertices);
	printf("  %u edges\n", gr->edges);
	printf("  %u triangles\n", gr->triangles);
	fflush(stdout);

	/****************************************************
	* Set up H-matrix
	****************************************************/

	bem_slp = new_slp_helmholtz_bem3d(k, gr, q_reg, q_sing, basis, basis);

	/* create cluster tree. */
	root = build_bem3d_cluster(bem_slp, clf, basis);

	/* create block tree. */
	broot = build_nonstrict_block(root, root, &eta, admissible_2_cluster);

	/* Set up interpolation approximation scheme for H-matrix V. */
	//setup_hmatrix_aprx_inter_row_bem3d(bem_slp, root, root, broot, m);
	setup_hmatrix_aprx_paca_bem3d(bem_slp, root, root, broot, accur);
	//setup_hmatrix_aprx_hca_bem3d(bem_slp, root, root, broot, m, accur);

	/****************************************************
	* Assemble H-matrix
	****************************************************/

	/* create h-matrix structure from block tree. */
	Z = build_from_block_hmatrix(broot, m * m * m);

	start_stopwatch(sw);
	/* assemble near- and farfield entries of v. */
	assemble_bem3d_hmatrix(bem_slp, broot, Z);
	t = stop_stopwatch(sw);
	/* get the total memory footprint for v. */
	size = getsize_hmatrix(Z) / 1024.0 / 1024.0;

	printf("H-matrix:\n");
	printf("  Assembly %.2f s\n", t);
	printf("  Storage %.3f mb\n", size);

	/****************************************************
	* Set up RHS
	****************************************************/

	x = new_avector(gr->vertices);
	random_avector(x);

	b = new_avector(gr->vertices);
	clear_avector(b);
	addevalsymm_hmatrix_avector(alpha, Z, x, b);

	//bval = 1;
	//fill_avector(b, bval);

	/****************************************************
	* Set up solver and decomposition parameters
	****************************************************/

	tm = new_releucl_truncmode();
	//tm->absolute = true;
	eps = 1e-12;
	maxiter = 1000;

	/****************************************************
	* Solve with H-matrix
	****************************************************/

	x_cg = new_zero_avector(gr->vertices);

	start_stopwatch(sw);
	solve_cg_hmatrix_avector(Z, b, x_cg, eps, maxiter);
	t = stop_stopwatch(sw);

	/* Calculate RMSE referenced to H-matrix solution */
	add_avector(-alpha, x, x_cg);
	rmse = norm2_avector(x_cg) / norm2_avector(x);

	printf("  Solve: %.2f s\n", t);
	printf("  Error: %.2f \n", rmse);
	fflush(stdout);

	/****************************************************
	* LU factorize H-matrix
	****************************************************/

	Z_lu = build_from_block_hmatrix(broot, m * m * m);
	x_lu = new_avector(gr->vertices);

	copy_hmatrix(Z, Z_lu);
    copy_avector(b, x_lu);

	start_stopwatch(sw);
	lrdecomp_hmatrix(Z_lu, tm, eps);
	t = stop_stopwatch(sw);
	size = getsize_hmatrix(Z_lu) / 1024.0 / 1024.0;

	printf("LU:\n");
	printf("  Factorization: %.2f s\n", t);
	printf("  Storage: %.3f mb\n", size);

	start_stopwatch(sw);
	lrsolve_hmatrix_avector(false, Z_lu, x_lu);
	t = stop_stopwatch(sw);

	/* Calculate RMSE referenced to H-matrix solution */
	add_avector(-alpha, x, x_lu);
	rmse = norm2_avector(x_lu) / norm2_avector(x);

	printf("  Solve: %.2f s\n", t);
	printf("  Error: %.2f \n", rmse);
	fflush(stdout);

	/****************************************************
	* Cholesky factorize H-matrix
	****************************************************/

	Z_chol = build_from_block_hmatrix(broot, m * m * m);
	x_chol = new_avector(gr->vertices);

	copy_hmatrix(Z, Z_chol);
	copy_avector(b, x_chol);

	start_stopwatch(sw);
	choldecomp_hmatrix(Z_chol, tm, eps);
	t = stop_stopwatch(sw);
	size = getsize_hmatrix(Z_chol) / 1024.0 / 1024.0;

	printf("Cholesky:\n");
	printf("  Factorization: %.2f s\n", t);
	printf("  Storage: %.3f mb\n", size);

	start_stopwatch(sw);
	cholsolve_hmatrix_avector(Z_chol, x_chol);
	t = stop_stopwatch(sw);

	/* Calculate RMSE referenced to H-matrix solution */
	add_avector(-alpha, x, x_chol);
	rmse = norm2_avector(x_chol) / norm2_avector(x);

	printf("  Solve: %.2f s\n", t);
	printf("  Error: %.2f \n", rmse);
	fflush(stdout);

	/****************************************************
	* Cleanup
	****************************************************/

	del_macrosurface3d(mg);
	del_surface3d(gr);
	freemem(root->idx);
	del_cluster(root);
	del_block(broot);
	del_hmatrix(Z);
	del_bem3d(bem_slp);
	del_stopwatch(sw);
	del_truncmode(tm);
	del_hmatrix(Z_lu);
	del_hmatrix(Z_chol);
	del_avector(b);
	del_avector(x);
	del_avector(x_cg);
	del_avector(x_lu);
	del_avector(x_chol);

	/* Uninit the H2Lib. */
	uninit_h2lib();

	return 0;
}