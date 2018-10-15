#include <stdio.h>

#include "basic.h"
#include "bem3d.h"
#include "helmholtzbem3d.h"
#include "clustergeometry.h"
#include "cluster.h"


typedef struct _zbem3d zbem3d;
typedef zbem3d * pzbem;


struct _zbem3d {
	real k;
	field kernel_const;
	real accur;
	real(*nodes)[3];
	uint nnodes;
	real dx, dy, dz;
	void (* nearfield)(const uint * ridx, const uint * cidx, pzbem zbem, bool ntrans, pamatrix N);
	void (* farfield_rk)(pccluster rc, pccluster cc, pzbem zbem, prkmatrix R);
};

pclustergeometry build_clustergeometry(pzbem zbem, uint ** idx);
pzbem new_zbem3d(real, real);
void del_zbem3d(pzbem);
inline field kernel(const real *, const real *, pzbem);
void nearfield(const uint *, const uint *, pzbem, bool, pamatrix);
void farfield_rk(pccluster, pccluster, pzbem, prkmatrix);
void assemble_block_hmatrix(uint, uint, uint, phmatrix *, pzbem);
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

  zbem->k = k;
  zbem->accur = accur;
  zbem->nodes = NULL;

  return zbem;
}


void del_zbem3d(pzbem zbem)
{
  freemem(zbem);
}


inline field kernel(const real * x, const real * y, pzbem zbem)
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


void nearfield(const uint * ridx, const uint * cidx, pzbem zbem, bool ntrans, pamatrix N)
{
  const real (* nodes)[3] = (const real(*)[3]) zbem->nodes;
  field * aa = N->a;
  uint rows = ntrans ? N->cols : N->rows;
  uint cols = ntrans ? N->rows : N->cols;
  longindex ld = N->ld;
  field kernel_const = zbem->kernel_const;
  real k = zbem->k;
  uint i, j;
  field sum;

  for (i = 0; i < rows; i++) {

	  sum = 0;

    for (j = 0; j < cols; j++) {
      sum += kernel(nodes[j + i * ld], nodes[j + i * ld], zbem);
    }

      if (ntrans) {
        aa[j + i * ld] = CONJ(sum) * kernel_const;
      }
      else {
        aa[i + j * ld] = sum * kernel_const;
      }
  }
}


void farfield_rk(pccluster rc, pccluster cc, pzbem zbem, prkmatrix R)
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


void assemble_block_hmatrix(uint bname, uint rname, uint cname, phmatrix * hn, pzbem zbem)
{
  phmatrix G = hn[bname];

  if (G->r) {
	zbem->farfield_rk(G->rc, rname, G->cc, cname, zbem, G->r);
  }
  else if (G->f) {
	zbem->nearfield(G->rc->idx, G->cc->idx, zbem, false, G->f);
  }
}


void assemble_hmatrix(pzbem zbem, pblock b, phmatrix G)
{
  phmatrix * hn = enumerate_hmatrix(b, G);

  iterate_byrow_block(b, 0, 0, 0, max_pardepth, NULL, assemble_block_hmatrix, zbem);

  freemem(hn);
}


int main(int argc, char **argv)
{
  pstopwatch sw;
  pclustergeometry cg;
  pcluster  root;
  pzbem zbem;
  uint clf;
  pblock broot;
  real eta, accur;
  phmatrix Z;
//  pavector gd, b, x;
  uint i, j;
  field k;
  real t, size, norm, dx, dy;
  uint nnodes, nnodes_x, nnodes_y;
  real (* nodes) [3];
  uint * idx;

  /* Init the H2Lib, should be called before any other function. */
  init_h2lib(&argc, &argv);

  /****************************************************
   * Set up basic parameters
   ****************************************************/

  clf = 4; /* Minimal leaf size for cluster tree construction. */
  eta = 1.4; /* Parameter 'eta' within the admissibilty condition. */
  k = 2 * M_PI * 1e6 / 1540.0; /* wavenumber */
  accur = 1e-3;
  nnodes_x = 10;
  nnodes_y = 10;
  dx = 4e-6;
  dy = 4e-6;
  sw = new_stopwatch(); /* Stopwatch for measuring the time. */

  /****************************************************
   * Create geometry
   ****************************************************/

  nnodes = nnodes_x * nnodes_y;
  nodes = (real (*) [3]) allocmem((size_t) sizeof(real * [3]) * nnodes);
  for (i = 0; i < nnodes_x; i++) {
    for (j = 0; j < nnodes_y; j++) {

      nodes[i + nnodes_y * j][0] =  i * dx;
      nodes[i + nnodes_y * j][1] = j * dy;
      nodes[i + nnodes_y * j][2] = 0;
    }
  }

  zbem = new_zbem3d(k, accur);
  zbem->nodes;
  zbem->nnodes;
  zbem->k;

  cg = build_clustergeometry(zbem, idx);

  /* Create cluster tree. */
  root = build_adaptive_cluster(cg, zbem->nnodes, idx, clf);

  /* Create block tree. */
  broot = build_nonstrict_block(root, root, &eta, admissible_2_cluster);

  /****************************************************
   * Assemble H-matrix
   ****************************************************/

  printf("Assemble H-matrix V:\n");

  /* Create H-matrix structure from block tree. */
  Z = build_from_block_hmatrix(broot, 2);

  start_stopwatch(sw);
  /* Assemble near- and farfield entries of V. */
  assemble_hmatrix(zbem, broot, Z);
  t = stop_stopwatch(sw);
  /* Get the total memory footprint for V. */
  size = getsize_hmatrix(Z) / 1024.0 / 1024.0;

  printf("  %.2f s\n", t);
  printf("  %.3f MB\n", size);


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
  del_block(broot);
  /* Permutation array for Dofs was automatically created by
   * 'build_bem3d_cluster', has to be free before the cluster tree. */
  freemem(root->idx);
  freemem(nodes);
  del_zbem3d(zbem);
  del_cluster(root);
  del_stopwatch(sw);

  /* Uninit the H2Lib. */
  uninit_h2lib();

  return 0;
}