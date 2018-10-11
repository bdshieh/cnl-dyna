

/****************************************************
 * This examples sets up single layer potential operator(SLP),
 * double layer potential operator(DLP) as well as the mass matrix M
 * as H-matrices in order to solve the interior Dirichlet
 * problem for the Laplace equation.
 ****************************************************/

pclustergeometry build_clustergeometry(real * nodes, uint nnodes, real dx, real dy, real dz, uint ** idx)
{
  pclustergeometry cg;
  uint i;

  cg = new_clustergeometry(3, nnodes);
  *idx = allocuint(nnodes);

  for (i = 0; i < nnodes; i++) {
    (*idx)[i] = i;

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


pbem3d new_bem3d(real k, real accur)
{
  pbem3d bem;

  bem->k = k;
  bem->accur = accur;
  bem->nearfield = ;
  bem->farfield_rk = assemble_bem3d_ACA_rkmatrix;


  return bem;
}


static void assemble_bem3d_block_hmatrix(pcblock b, uint bname, uint rname, uint cname, uint pardepth, void *data)
{
  pbem3d    bem = (pbem3d) data;
  paprxbem3d aprx = bem->aprx;
  pparbem3d par = bem->par;
  phmatrix *hn = par->hn;
  phmatrix  G = hn[bname];

  (void) b;
  (void) pardepth;

  if (G->r) {
    bem->farfield_rk(G->rc, rname, G->cc, cname, bem, G->r);
    if (aprx->recomp == true) {
      trunc_rkmatrix(0, aprx->accur_recomp, G->r);
    }
  }
  else if (G->f) {
    bem->nearfield(G->rc->idx, G->cc->idx, bem, false, G->f);
  }
}


int main(int argc, char **argv)
{
  pstopwatch sw;
  pclustergeometry cg;
  pcluster  root;
  pbem3d bem;
  uint clf;
  pblock broot;
  real eta;
  phmatrix V;
  pavector gd, b, x;
  uint m;
  real eps_solve;
  uint maxiter;
  real t, size, norm;


  /* Init the H2Lib, should be called before any other function. */
  init_h2lib(&argc, &argv);

  /****************************************************
   * Set up basic parameters
   ****************************************************/

  /* Number of interpolation points */
  m = 4;

  /* Minimal leaf size for cluster tree construction. */
  clf = 2 * m * m * m;

  /* Parameter 'eta' within the admissibilty condition. */
  eta = 1.4;

  /* absolute norm of the residuum for CG-method */
  eps_solve = 1.0e-10;

  /* maximum number of CG-steps that should be performed. */
  maxiter = 500;

  /* Stopwatch for measuring the time. */
  sw = new_stopwatch();

  /****************************************************
   * Create geometry
   ****************************************************/

   cg = build_clustergeometry();

  /****************************************************
   * Set up basis data structures for H-matrix approximations
   ****************************************************/

  bem = new_bem3d()

  /* Create cluster tree. */
  root = build_adaptive_cluster(cg, nnodes, idx, clf);

  /* Create block tree. */
  broot = build_nonstrict_block(root, root, &eta, admissible_2_cluster);

  /* Set up interpolation approximation scheme for H-matrix V. */
  setup_hmatrix_aprx_aca_bem3d(bem, root, root, broot, m);


  /****************************************************
   * Assemble H-matrix
   ****************************************************/

  printf("Assemble H-matrix V:\n");

  /* Create H-matrix structure from block tree. */
  V = build_from_block_hmatrix(broot, m * m * m);

  start_stopwatch(sw);
  /* Assemble near- and farfield entries of V. */
  assemble_bem3d_hmatrix(bem, broot, V);
  t = stop_stopwatch(sw);
  /* Get the total memory footprint for V. */
  size = getsize_hmatrix(V) / 1024.0 / 1024.0;

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
  del_hmatrix(V);
  del_hmatrix(KM);
  del_block(broot);
  /* Permutation array for Dofs was automatically created by
   * 'build_bem3d_cluster', has to be free before the cluster tree. */
  freemem(root->idx);
  del_cluster(root);
  del_stopwatch(sw);

  /* Uninit the H2Lib. */
  uninit_h2lib();

  return 0;
}