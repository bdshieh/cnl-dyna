#include <stdio.h>

#include "basic.h"
#include "bem3d.h"
#include "helmholtzbem3d.h"
#include "clustergeometry.h"
#include "cluster.h"


typedef struct _zbem3d zbem3d;
typedef zbem3d * pzbem;


struct _zbem3d {
	real k, accur;
	field kernel_const;
	real (*nodes) [3];
	uint nnodes;
	real dx, dy, dz;
	//void (* nearfield)(const uint * ridx, const uint * cidx, pzbem zbem, bool ntrans, pamatrix N);
	//void (* farfield_rk)(pccluster rc, pccluster cc, pzbem zbem, prkmatrix R);
	phmatrix * hn;
};

pclustergeometry build_clustergeometry(pzbem, uint **);
pzbem new_zbem3d(real, real);
void del_zbem3d(pzbem);
field kernel(const real *, const real *, pzbem);
void nearfield(const uint *, const uint *, void *, bool, pamatrix);
void farfield_rk_aca(pccluster, pccluster, pzbem, prkmatrix);
void farfield_rk_paca(pccluster, pccluster, pzbem, prkmatrix);
void assemble_block_hmatrix(pcblock, uint, uint, uint, uint, void *);
void assemble_hmatrix(pzbem, pblock, phmatrix);


pclustergeometry build_clustergeometry(pzbem zbem, uint ** idx)
{
  pclustergeometry cg;
  real (* nodes)[3] = zbem->nodes;
  uint nnodes = zbem->nnodes;
  real dx = zbem->dx;
  real dy = zbem->dy;
  real dz = zbem->dz;
  uint i;

  cg = new_clustergeometry(3, nnodes);
  * idx = allocuint(nnodes);

  for (i = 0; i < nnodes; i++) {
	(* idx)[i] = i;

	/* Center of gravity as characteristic point */
	cg->x[i][0] = nodes[i][0];
	cg->x[i][1] = nodes[i][1];
	cg->x[i][2] = nodes[i][2];

	/* Lower front left corner of bounding box */
	cg->smin[i][0] = nodes[i][0] - dx / 2;
	cg->smin[i][1] = nodes[i][1] - dy / 2;
	cg->smin[i][2] = nodes[i][2] - dz / 2;

	/* Upper back right corner of bounding box */
	cg->smax[i][0] = nodes[i][0] + dx / 2;
	cg->smax[i][1] = nodes[i][1] + dy / 2;
	cg->smax[i][2] = nodes[i][2] + dz / 2;

	cg->w[i] = 1;
  }

  return cg;
}


pzbem new_zbem3d(real k, real accur)
{
  pzbem zbem;

  zbem = (pzbem)allocmem(sizeof(zbem3d));

  zbem->k = k;
  zbem->accur = accur;
  zbem->nodes = NULL;
  zbem->hn = NULL;
  //zbem->nearfield = NULL;
  //zbem->farfield_rk = NULL;

  return zbem;
}


void del_zbem3d(pzbem zbem)
{
  freemem(zbem);
}


field kernel(const real * x, const real * y, pzbem zbem)
{
  field kernel_const = zbem->kernel_const;
  real k_real = REAL(zbem->k);
  real k_imag = -IMAG(zbem->k);
  real dist[3];
  real norm, norm2, rnorm;

  field res;

  dist[0] = x[0] - y[0];
  dist[1] = x[1] - y[1];
  dist[2] = x[2] - y[2];
  norm2 = REAL_NORMSQR3(dist[0], dist[1], dist[2]);
  rnorm = REAL_RSQRT(norm2);

  norm = norm2 * rnorm;
  if (k_imag != 0.0) {
	rnorm *= REAL_EXP(k_imag * norm);
  }
  norm = k_real * norm;
  res = kernel_const * (rnorm * REAL_COS(norm)) + (REAL_SIN(norm) * rnorm) * I;

  return res;
}


void nearfield(const uint * ridx, const uint * cidx, void * data, bool ntrans, pamatrix N)
{
  pzbem zbem = (pzbem) data;
  const real (* nodes)[3] = (const real(*)[3]) zbem->nodes;
  field * aa = N->a;
  uint rows = ntrans ? N->cols : N->rows;
  uint cols = ntrans ? N->rows : N->cols;
  longindex ld = N->ld;
  field kernel_const = zbem->kernel_const;
  real k = zbem->k;
  uint i, j;

  for (i = 0; i < rows; i++) {

	for (j = 0; j < cols; j++) {

		if (ntrans) {
			aa[j + i * ld] = kernel_const * kernel(nodes[ridx[i]], nodes[cidx[j]], zbem);;
		}
		else {
			aa[i + j * ld] = kernel_const * kernel(nodes[ridx[i]], nodes[cidx[j]], zbem);;
		}
	}
  }
}


void farfield_rk_aca(pccluster rc, pccluster cc, pzbem zbem, prkmatrix R)
{
  const real accur = zbem->accur;
  const uint * ridx = rc->idx;
  const uint * cidx = cc->idx;
  const uint rows = rc->size;
  const uint cols = cc->size;

  pamatrix  G;

  G = new_amatrix(rows, cols);
  nearfield(ridx, cidx, zbem, false, G);
  decomp_fullaca_rkmatrix(G, accur, NULL, NULL, R);

  del_amatrix(G);
}


void farfield_rk_paca(pccluster rc, pccluster cc, pzbem zbem, prkmatrix R)
{
	const uint * ridx = rc->idx;
	const uint * cidx = cc->idx;
	const uint rows = rc->size;
	const uint cols = cc->size;
	const real accur = zbem->accur;
	matrixentry_t entry = (matrixentry_t) nearfield;

	decomp_partialaca_rkmatrix(entry, (void *) zbem, ridx, rows, cidx, cols, accur, NULL, NULL, R);
}


void assemble_block_hmatrix(pcblock b, uint bname, uint rname, uint cname, uint par_depth, void * data)
{ 
  pzbem zbem = (pzbem) data;
  phmatrix * hn = zbem->hn;
  phmatrix G = hn[bname];

  //printf("block %u \n", bname);
  (void) b;
  (void) rname;
  (void) cname;
  (void) par_depth;

  
  if (G->r) {
	farfield_rk_paca(G->rc, G->cc, zbem, G->r);
  }
  else if (G->f) {
    nearfield(G->rc->idx, G->cc->idx, (void *) zbem, false, G->f);
  }
}


void assemble_hmatrix(pzbem zbem, pblock b, phmatrix G)
{
  //phmatrix * hn = enumerate_hmatrix(b, G);
  zbem->hn = enumerate_hmatrix(b, G);

  iterate_byrow_block(b, 0, 0, 0, max_pardepth, NULL, assemble_block_hmatrix, zbem);

  freemem(zbem->hn);
  zbem->hn = NULL;
}


int main(int argc, char **argv)
{
  pstopwatch sw;
  pclustergeometry cg;
  pcluster  root;
  pzbem zbem;
  pblock broot;
  phmatrix Z;
//  pavector gd, b, x;
  uint clf;
  uint i, j;
  field kernel_const;
  real k, eta, accur;
  real t, size, norm, dx, dy, dz;
  uint nnodes, nnodes_x, nnodes_y;
  real (* nodes) [3];
  uint * idx;

  /* Init the H2Lib, should be called before any other function. */
  init_h2lib(&argc, &argv);

  /****************************************************
   * Set up basic parameters
   ****************************************************/

  clf = 10; /* Minimal leaf size for cluster tree construction. */
  eta = 2; /* Parameter 'eta' within the admissibilty condition. */
  k = 2 * M_PI * 1e6 / 1540.0; /* wavenumber */
  accur = 1e-2;
  nnodes_x = 316;
  nnodes_y = 316;
  dx = 4e-6;
  dy = 4e-6;
  dz = 1e-6;
  sw = new_stopwatch(); /* Stopwatch for measuring the time. */

  /****************************************************
   * Create geometry
   ****************************************************/

  nnodes = nnodes_x * nnodes_y;
  nodes = (real (*) [3]) allocmem((size_t) sizeof(real * [3]) * nnodes);
  for (i = 0; i < nnodes_x; i++) {
    for (j = 0; j < nnodes_y; j++) {

      nodes[j + nnodes_y * i][0] =  i * dx + dx / 2;
      nodes[j + nnodes_y * i][1] = j * dy + dy / 2;
      nodes[j + nnodes_y * i][2] = 0;
    }
  }

  //for (uint i = 0; i < nnodes; i++) { printf("%.1e %.1e %.1e \n", nodes[i][0], nodes[i][1], nodes[i][2]); }

  zbem = new_zbem3d(k, accur);
  zbem->nodes = nodes;
  zbem->nnodes = nnodes;
  zbem->k = k;
  zbem->dx = dx;
  zbem->dy = dy;
  zbem->dz = dz;
  zbem->accur = accur;
  zbem->kernel_const = 1;
  //zbem->farfield_rk = farfield_rk;
  //zbem->nearfield = nearfield;

  cg = build_clustergeometry(zbem, &idx);
  //printf("  %u \n", cg->dim);
  //printf("  %u \n", cg->nidx);
  //for (uint k = 0; k < nnodes; k++) { printf("%u \n", idx[k]); }

  /* create cluster tree. */
  root = build_adaptive_cluster(cg, zbem->nnodes, idx, clf);
  //root = build_regular_cluster(cg, zbem->nnodes, idx, clf, 0);
  printf("  %u \n", root->size);
  printf("  %u \n", root->sons);
  printf("  %u \n", root->dim);
  printf("  %u \n", root->desc);

  /* create block tree. */
  broot = build_nonstrict_block(root, root, &eta, admissible_2_cluster);
  printf("created blocktree \n");
  /****************************************************
   * Assemble H-matrix
   ****************************************************/

  //printf("assemble h-matrix v:\n");

  ///* create h-matrix structure from block tree. */
  Z = build_from_block_hmatrix(broot, 4);
  printf("  %u \n", Z->desc);
  printf("  %u \n", Z->rsons);
  printf("  %u \n", Z->csons);

  start_stopwatch(sw);
  /* assemble near- and farfield entries of v. */
  assemble_hmatrix(zbem, broot, Z);
  t = stop_stopwatch(sw);
  /* get the total memory footprint for v. */
  size = getsize_hmatrix(Z) / 1024.0 / 1024.0;

  printf("  %.2f s\n", t);
  printf("  %.3f mb\n", size);


  /****************************************************
   * Compute right-hand-side b = (0.5M + K)*gd
   ****************************************************/

  /* Create new vector to store right-hand-side. */
//  b = new_avector(gr->triangles);
//  printf("Compute right-hand-side:\n");
//  start_stopwatch(sw);
//  clear_avector(b);
//  /* H-matrix vector product. */
//  addeval_hmatrix_avector(1.0, KM, gd, b);
//  t = stop_stopwatch(sw);
//  size = getsize_avector(b) / 1024.0 / 1024.0;
//  printf("  %.2f s\n", t);
//  printf("  %.3f MB\n", size);

  /****************************************************
   * Solve linear system V x = b using CG-method.
   ****************************************************/

  /* Create new vector to store the solution coefficients. */
//  x = new_avector(gr->triangles);
//  printf("Solve linear system:\n");
//  start_stopwatch(sw);
//  /* Call the CG-solver for H-matrices. */
//  solve_cg_hmatrix_avector(V, b, x, eps_solve, maxiter);
//  t = stop_stopwatch(sw);
//  size = getsize_avector(x) / 1024.0 / 1024.0;
//  printf("  %.2f s\n", t);
//  printf("  %.3f MB\n", size);


  /****************************************************
   * cleanup
   ****************************************************/

//  del_avector(x);
//  del_avector(b);
//  del_avector(gd);
  del_hmatrix(Z);
  del_clustergeometry(cg);
  ///* Permutation array for Dofs was automatically created by
  // * 'build_bem3d_cluster', has to be free before the cluster tree. */
  freemem(root->idx);
  del_cluster(root);
  del_block(broot);
  del_zbem3d(zbem);
  freemem(nodes);
  del_stopwatch(sw);

  /* Uninit the H2Lib. */
  uninit_h2lib();

  return 0;
}