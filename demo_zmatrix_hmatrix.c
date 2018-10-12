

/****************************************************
 * This examples sets up single layer potential operator(SLP),
 * double layer potential operator(DLP) as well as the mass matrix M
 * as H-matrices in order to solve the interior Dirichlet
 * problem for the Laplace equation.
 ****************************************************/

pclustergeometry build_clustergeometry(pzbem zbem, uint ** idx)
{
  pclustergeometry cg;
  real (* nodes)[3] = pzbem->nodes;
  uint nnodes = pzbem->nnodes;
  real dx = pzbem->dx;
  real dy = pzbem->dy;
  real dz = pzbem->dz;
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
  zbem->par = new_parbem3d();

  return zbem;
}


void del_zbem3d(pzbem zbem)
{
  del_parbem3d(bem->par);
  freemem(zbem);
}


static inline field kernel(const real * x, const real * y, pzbem zbem)
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


static void nearfield(const uint * ridx, const uint * cidx, pzbem zbem, bool ntrans, pamatrix N)
{
  const real (* nodes)[3] = (const real(*)[3]) zbem->nodes;
  field * aa = N->a;
  uint rows = ntrans ? N->cols : N->rows;
  uint cols = ntrans ? N->rows : N->cols;
  longindex ld = N->ld;
  field kernel_const = zbem->kernel_const
  field k = zbem->k
  uint i, j;
  field sum;

  for (i = 0; i < rows; i++) {

    sum = 0

    for (j = 0; j < cols; j++) {
      sum += kernel(nodes[i][j][0], nodes[i][j][1], k, kernel_const);
    }

      if (ntrans) {
        aa[j + i * ld] = CONJ(sum) * kernel_const;
      }
      else {
        aa[i + j * ld] = sum * kernel_const;
      }
  }
}


static void farfield_rk(pccluster rc, pccluster cc, pzbem zbem, prkmatrix R)
{
  const real accur = zbem->accur;
  const uint * ridx = rc->idx;
  const uint * cidx = cc->idx;
  const uint rows = rc->size;
  const uint cols = cc->size;

  pamatrix  G;

  G = new_amatrix(rows, cols);
  nearfield(ridx, cidx, bem, false, G);
  decomp_fullaca_rkmatrix(G, accur, NULL, NULL, R);

  del_amatrix(G);
}


static void assemble_block_hmatrix(uint bname, uint rname, uint cname, void * data)
{
  pzbem zbem = (pzbem) data;
  pparbem3d par = zbem->par;
  phmatrix * hn = par->hn;
  phmatrix G = hn[bname];

  if (G->r) {
	bem->farfield_rk(G->rc, rname, G->cc, cname, bem, G->r);
  }
  else if (G->f) {
	bem->nearfield(G->rc->idx, G->cc->idx, bem, false, G->f);
  }
}


void assemble_hmatrix(pzbem zbem, pblock b, phmatrix G)
{
  pparbem3d par = zbem->par;
  par->hn = enumerate_hmatrix(b, G);

  iterate_byrow_block(b, 0, 0, 0, max_pardepth, NULL, assemble_block_hmatrix, bem);

  freemem(par->hn);
  par->hn = NULL;
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
  pavector gd, b, x;
  uint m;
  field k;
  real t, size, norm;


  /* Init the H2Lib, should be called before any other function. */
  init_h2lib(&argc, &argv);

  /****************************************************
   * Set up basic parameters
   ****************************************************/

  m = 4; /* Number of interpolation points */
  clf = 2 * m * m * m; /* Minimal leaf size for cluster tree construction. */
  eta = 1.4; /* Parameter 'eta' within the admissibilty condition. */
  k = 2 * M_PI * 1e6 / 1540.0; /* wavenumber */
  accur = 1e-3;
  sw = new_stopwatch(); /* Stopwatch for measuring the time. */

  /****************************************************
   * Create geometry
   ****************************************************/

  zbem = new_zbem3d(k, accur)
  zbem->nodes;
  zbem->nnodes;
  zbem->k;
  idx;

  cg = build_clustergeometry(zbem, idx);

  /* Create cluster tree. */
  root = build_adaptive_cluster(cg, zbem->nnodes, idx, clf);

  /* Create block tree. */
  broot = build_nonstrict_block(root, root, &eta, admissible_2_cluster);

  /* Set up interpolation approximation scheme for H-matrix V. */
  setup_hmatrix_aprx_aca_bem3d(bem, root, root, broot, m);


  /****************************************************
   * Assemble H-matrix
   ****************************************************/

  printf("Assemble H-matrix V:\n");

  /* Create H-matrix structure from block tree. */
  Z = build_from_block_hmatrix(broot, m * m * m);

  start_stopwatch(sw);
  /* Assemble near- and farfield entries of V. */
  assemble_bem3d_hmatrix(bem, broot, Z);
  t = stop_stopwatch(sw);
  /* Get the total memory footprint for V. */
  size = getsize_hmatrix(Z) / 1024.0 / 1024.0;

  printf("  %.2f s\n", t);
  printf("  %.3f MB\n", size);


  /****************************************************
   * Compute right-hand-side b = (0.5M + K)*gd
   ****************************************************/

  /* Create new vector to store right-hand-side. */
  b = new_avector(gr->triangles);
  printf("Compute right-hand-side:\n");
  start_stopwatch(sw);
  clear_avector(b);
  /* H-matrix vector product. */
  addeval_hmatrix_avector(1.0, KM, gd, b);
  t = stop_stopwatch(sw);
  size = getsize_avector(b) / 1024.0 / 1024.0;
  printf("  %.2f s\n", t);
  printf("  %.3f MB\n", size);

  /****************************************************
   * Solve linear system V x = b using CG-method.
   ****************************************************/

  /* Create new vector to store the solution coefficients. */
  x = new_avector(gr->triangles);
  printf("Solve linear system:\n");
  start_stopwatch(sw);
  /* Call the CG-solver for H-matrices. */
  solve_cg_hmatrix_avector(V, b, x, eps_solve, maxiter);
  t = stop_stopwatch(sw);
  size = getsize_avector(x) / 1024.0 / 1024.0;
  printf("  %.2f s\n", t);
  printf("  %.3f MB\n", size);


  /****************************************************
   * cleanup
   ****************************************************/

  del_avector(x);
  del_avector(b);
  del_avector(gd);
  del_hmatrix(Z);
  del_block(broot);
  /* Permutation array for Dofs was automatically created by
   * 'build_bem3d_cluster', has to be free before the cluster tree. */
  freemem(root->idx);
  del_zbem3d(zbem);
  del_cluster(root);
  del_stopwatch(sw);

  /* Uninit the H2Lib. */
  uninit_h2lib();

  return 0;
}