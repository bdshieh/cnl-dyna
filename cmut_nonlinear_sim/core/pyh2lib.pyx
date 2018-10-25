

from . h2lib cimport *


cpdef class Macrosurface3d:

    cdef pmacrosurface3d _c_pmacrosurface3d
    cdef readonly uint * vertices
    cdef readonly uint * edges
    cdef readonly uint * triangles
    cdef public real [:, :] x
    cdef public uint [:, :] e
    cdef public uint [:, :] t
    cdef public uint [:, :] s

    def __cinit__(self, uint vertices=None, uint edges=None, uint triangles=None, pstruct=None):

        if pstruct is not None:
            ms = <pmacrosurface3d> pstruct
        else:
            ms = new_macrosurface3d(vertices, edges, triangles)

        self.vertices = ms.vertices
        self.edges = ms.edges
        self.triangles = ms.triangles

        self.x = ms.x
        self.e = ms.edges
        self.t = ms.t
        self.s = ms.s

        ms.phi = cube_parametrization
        ms.phidata = ms

        self._c_pmacrosurface3d = ms

    def __dealloc(self):
        del_macrosurface3d(self._c_pmacrosurface3d)

    @property
    def vertices(self):
        return self.vertices[0]

    @property
    def edges(self):
        return self.edges[0]

    @property
    def triangles(self):
        return self.triangles[0]


cpdef class Surface3d:

    cdef psurface3d _c_psurface3d
    cdef readonly uint * vertices
    cdef readonly uint * edges
    cdef readonly uint * triangles
    cdef public real [:, :] x
    cdef public uint [:, :] e
    cdef public uint [:, :] t
    cdef public uint [:, :] s
    cdef public real [:, :] n

    def __cinit__(self, uint vertices=None, uint edges=None, uint triangles=None, pstruct=None):

        if pstruct is not None:
            surf = <psurface3d> pstruct
        else:
            surf = new_surface3d(vertices, edges, triangles)

        self.vertices = &surf.vertices
        self.edges = &surf.edges
        self.triangles = &surf.triangles

        self.x = surf.x
        self.e = surf.e
        self.t = surf.t
        self.s = surf.s
        self.n = surf.n

        self._c_pmacrosurface3d = surf

    def __dealloc(self):
        del_surface3d(self._c_psurface3d)
    
    @property
    def vertices(self):
        return self.vertices[0]

    @property
    def edges(self):
        return self.edges[0]

    @property
    def triangles(self):
        return self.triangles[0]


cpdef class Bem3d:

    cdef pbem3d _c_pbem3d
    cdef public field * k
    cdef public field * kernel_const
    cdef public basisfunctionbem3d * row_basis
    cdef public basisfunctionbem3d * col_basis

    def __cinit__(self, pcsurface3d gr=None, field k=None, field kernel_const=None, basisfunctionbem3d row_basis=None, 
        basisfunctionbem3d col_basis=None, pstruct=None):

        if pstruct is not None:
            bem = <pbem3d> pstruct
        else:
            bem = new_bem3d(pcsurface3d gr, basisfunctionbem3d row_basis, basisfunctionbem3d col_basis)

        self.k = bem.k
        self.kernel_const = bem.kernel_const
        self.row_basis = bem.row_basis
        self.col_basis = bem.col_basis

        self._c_pbem3d = bem

    def __dealloc(self):
        del_bem3d(self._c_bem3d)

    @property
    def k(self):
        return self.k[0]

    @k.setter
    def k(self, field val):
        self.k[0] = val

    @property
    def kernel_const(self):
        return self.kernel_const[0]

    @k.setter
    def kernel_const(self, field val):
        self.kernel_const[0] = val

    @property
    def row_basis(self):
        return self.row_basis[0]

    @k.setter
    def row_basis(self, basisfunctionbem3d val):
        self.row_basis[0] = val

    @property
    def col_basis(self):
        return self.col_basis[0]

    @k.setter
    def col_basis(self, basisfunctionbem3d val):
        self.col_basis[0] = val


cpdef class Cluster:

    cdef pcluster _c_pcluster
    cdef readonly uint [:] idx
    cdef readonly uint * sons
    cdef readonly uint * dim
    cdef readonly uint * desc

    def __cinit__(self, uint size=None, uint [:] idx=None, uint sons=None, uint dim=None, pstruct=None):

        if pstruct is not None:
            cl = <pcluster> pstruct
        else:
            cl = new_cluster(size, idx, sons, dim)

        self.idx = cl.idx
        self.sons = &cl.sons
        self.dim = &cl.dim
        self.desc = &cl.desc

        self._c_pcluster = cl

    def __dealloc(self):
        del_bem3d(self._c_pcluster)

    @property
    def sons(self):
        return self.sons[0]

    @property
    def dim(self):
        return self.dim[0]

    @property
    def desc(self):
        return self.desc[0]


cpdef class Block:

    cdef pblock _c_pblock
    cdef readonly bool * a
    cdef readonly uint * rsons
    cdef readonly uint * csons
    cdef readonly uint * desc

    def __cinit__(self, pcluster rc=None, pcluster cc=None, bool a=None, uint rson=None, uint csons=None, 
            row_basis=None, pstruct=None):

        if pobj is not None:
            bl = <pblock> pobj
        else:
            bl = new_block(rc, cc, a, rsons, csons)

        self.a = &bl.a
        self.rsons = &bl.rsons
        self.csons = &bl.csons
        self.desc = &bl.desc

        self._c_pblock = bl

    def __dealloc(self):
        del_bem3d(self._c_pblock)


cpdef class Hmatrix:

    cdef pbem3d _c_pbem3d
    cdef public field k
    cdef public field kernel_const
    cdef public basisfunctionbem3d row_basis
    cdef public basisfunctionbem3d col_basis

    def __cinit__(self, field k=None, field kernel_const=None, basisfunctionbem3d row_basis=None, 
        basisfunctionbem3d col_basis=None, pobj=None):

        if pobj is not None:
            bem = <pbem3d> pobj
        else:
            bem = new_bem3d(vertices, edges, triangles)

        self.k = bem.k
        self.kernel_const = bem.kernel_const
        self.row_basis = bem.row_basis
        self.col_basis = bem.col_basis

        self._c_pbem3d = bem

    def __dealloc(self):
        del_bem3d(self._c_bem3d)


cpdef build_from_macrosurface3d_surface3d(Macrosurface3d ms, uint refn):

    surf = build_from_macrosurface3d_surface3d(ms->_c_pmacrosurface3d, refn)
    return Surface3d(pobj=surf)


cpdef new_slp_helmholtz_bem3d(field k, Surface3D gr, uint q_reg, uint q_sing, basisfunctionbem3d basis,  
        basisfunctionbem3d basis):
    pass


cpdef build_bem3d_cluster(Bem3d bem_slp, uint clf, basisfunctionbem3d basis):
    pass


cpdef build_nonstrict_block(Block root, Block root, real &eta, admissible admissible_2_cluster):
    pass