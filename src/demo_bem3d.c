#include <stdio.h>

#include "basic.h"
#include "macrosurface3d.h"
#include "helmholtzbem3d.h"

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
	phmatrix Z;

	/* Init the H2Lib, should be called before any other function. */
	init_h2lib(&argc, &argv);

	/****************************************************
	* Set up basic parameters
	****************************************************/

	m = 4;
	clf = 2 * m * m * m; /* Minimal leaf size for cluster tree construction. */
	eta = 2; /* Parameter 'eta' within the admissibilty condition. */
	k = 2 * M_PI * 1e6 / 1540.0; /* wavenumber */
	accur = 1e-2; /* accuracy of rk as a fraction of the norm */
	ax = 100e-6;
	bx = 100e-6;
	refn = 64;
	q_reg = 2;
	q_sing = q_reg + 2;
	basis = BASIS_LINEAR_BEM3D;

	sw = new_stopwatch(); /* Stopwatch for measuring the time. */

	/****************************************************
	* Create geometry
	****************************************************/

	mg = new_square_macrosurface3d(ax, bx);
	gr = build_from_macrosurface3d_surface3d(mg, refn);
	printf("%u vertices \n", gr->vertices);
	printf("%u edges \n", gr->edges);
	printf("%u triangles \n", gr->triangles);

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

	printf("  %.2f s\n", t);
	printf("  %.3f mb\n", size);


	/****************************************************
	* cleanup
	****************************************************/

	del_macrosurface3d(mg);
	del_surface3d(gr);
	freemem(root->idx);
	del_cluster(root);
	del_block(broot);
	del_hmatrix(Z);
	del_bem3d(bem_slp);
	del_stopwatch(sw);

	/* Uninit the H2Lib. */
	uninit_h2lib();

	return 0;
}