#include <stdio.h>

#include "basic.h"
#include "bem3d.h"
#include "helmholtzbem3d.h"
#include "clustergeometry.h"


pclustergeometry build_clustergeometry(pzbem, uint **);
pzbem new_zbem3d(real, real);
void del_zbem3d(pzbem);
static inline field kernel(const real *, const real *, pzbem);
static void nearfield(const uint *, const uint *, pzbem, bool, pamatrix);
static void farfield_rk(pccluster, pccluster, pzbem, prkmatrix) ;
static void assemble_block_hmatrix(uint, uint, uint, void *);
void assemble_hmatrix(pzbem, pblock, phmatrix);


typedef struct zbem3d {
  field k;
  field kernel_const;
  real accur;
  real (* nodes)[3];
  uint nnodes;
  real dx;
  real dy;
  real dz;
  pparbem3d par;
};

typedef * zbem3d pzbem;