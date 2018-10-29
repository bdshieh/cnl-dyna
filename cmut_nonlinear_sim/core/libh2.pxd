

## complex.h ##
cdef extern from 'complex.h' nogil:

    double cabs(double complex)
    double carg(double complex)
    double complex conj(double complex)
    double complex cexp(double complex)
    double creal(double complex)
    double cimag(double complex)

## basic types ##
ctypedef double real
ctypedef double complex field
ctypedef unsigned int uint
ctypedef unsigned long size_t

## basic.h ##
cdef extern from 'basic.h':

    void init_h2lib(int *argc, char ***argv)
    void uninit_h2lib()


## vector.h ##
cdef extern from 'avector.h' nogil:
    
    cdef struct _avector:
        field * v
        uint dim

    ctypedef _avector avector
    ctypedef avector * pavector
    ctypedef const avector * pcavector

    pavector new_avector(uint dim)
    pavector new_zero_avector(uint dim)
    void del_avector(pavector v)
    void add_avector(field alpha, pcavector x, pavector y)
    void clear_avector(pavector v)
    void fill_avector(pavector v, field x)
    void scale_avector(field alpha, pavector v)
    void copy_avector(pavector v, pavector w)
    void random_avector(pavector v)
    real norm2_avector(pcavector v)


## amatrix.h ##
cdef extern from 'amatrix.h' nogil:

    cdef struct _amatrix:
        field * a
        uint ld
        uint rows
        uint cols
        
    ctypedef _amatrix amatrix
    ctypedef amatrix * pamatrix
    ctypedef const amatrix * pcamatrix

    pamatrix new_amatrix(uint rows, uint cols)
    pamatrix new_zero_amatrix(uint rows, uint cols)
    pamatrix new_identity_amatrix(uint rows, uint cols)
    void del_amatrix(pamatrix a)
    void clear_amatrix(pamatrix a)
    void identity_amatrix(pamatrix a)
    void random_amatrix (pamatrix a)
    void copy_amatrix(bint atrans, pcamatrix a, pamatrix b)
    pamatrix clone_amatrix(pcamatrix src)
    void scale_amatrix(field alpha, pamatrix a)
    void conjugate_amatrix(pamatrix a)
    real norm2_amatrix(pcamatrix a)
    real norm2diff_amatrix(pcamatrix a, pcamatrix b)
    void addeval_amatrix_avector(field alpha, pcamatrix a, pcavector, src, pavector trg)
    void add_amatrix(field alpha, bint atrans, pcamatrix a, pamatrix b)


## surface3d.h ##
cdef extern from 'surface3d.h' nogil:
    
    cdef struct _surface3d:
        uint vertices
        uint edges
        uint triangles
        real (* x)[3]
        uint (* e)[2]
        uint (* t)[3]
        uint (* s)[3]
        real (* n)[3]

    ctypedef _surface3d surface3d
    ctypedef surface3d * psurface3d
    ctypedef const surface3d * pcsurface3d

    psurface3d new_surface3d(uint vertices, uint edges, uint triangles)
    void del_surface3d(psurface3d gr)


## macrosurface3d.h ##
cdef extern from 'macrosurface3d.h' nogil:

    cdef struct _macrosurface3d:
        uint vertices
        uint edges
        uint triangles
        real (* x)[3]
        uint (* e)[2]
        uint (* t)[3]
        uint (* s)[3]
        void (* phi) (uint i, real xr1, real xr2, void * phidata, real xt[3])
        void * phidata

    ctypedef _macrosurface3d macrosurface3d
    ctypedef macrosurface3d * pmacrosurface3d
    ctypedef const macrosurface3d * pcmacrosurface3d

    pmacrosurface3d new_macrosurface3d(uint vertices, uint edges, uint triangles)
    void del_macrosurface3d(pmacrosurface3d mg)
    psurface3d build_from_macrosurface3d_surface3d(pcmacrosurface3d mg, uint split) 


## cluster.h ##
cdef extern from 'cluster.h' nogil:
    
    cdef struct _cluster:
        uint size
        uint * idx
        uint sons
        uint dim
        real * bmin
        real * bmax
        uint desc
        uint type

    ctypedef _cluster cluster
    ctypedef cluster * pcluster
    ctypedef const cluster * pccluster

    pcluster new_cluster(uint size, uint * idx, uint sons, uint dim)
    void del_cluster(pcluster t)


## block.h ##
cdef extern from 'block.h' nogil:

    cdef struct _block:
        pcluster rc
        pcluster cc
        bint a
        uint rsons
        uint csons
        uint desc
        
    ctypedef _block block
    ctypedef block * pblock
    ctypedef const block * pcblock
    ctypedef bint (* admissible) (pcluster rc, pcluster cc, void * data)

    pblock new_block(pcluster rc, pcluster cc, bint a, uint rsons, uint csons)
    void del_block(pblock b)
    pblock build_nonstrict_block(pcluster rc, pcluster cc, void * data, admissible admis)


## rkmatrix.h ##
cdef extern from 'rkmatrix.h' nogil:

    cdef struct _rkmatrix:
        amatrix A
        amatrix B
        uint k
    
    ctypedef _rkmatrix rkmatrix
    ctypedef rkmatrix * prkmatrix
    ctypedef const rkmatrix * pcrkmatrix

    prkmatrix new_rkmatrix(uint rows, uint cols, uint k)
    void del_rkmatrix(prkmatrix r)


## hmatrix.h ##
cdef extern from 'hmatrix.h' nogil:

    cdef struct _hmatrix

    ctypedef _hmatrix hmatrix
    ctypedef hmatrix * phmatrix
    ctypedef const hmatrix * pchmatrix

    cdef struct _hmatrix:
        pccluster rc
        pccluster cc
        prkmatrix r
        pamatrix f
        phmatrix * son
        uint rsons
        uint csons
        uint desc
        
    phmatrix new_hmatrix(pccluster rc, pccluster cc)
    phmatrix new_super_hmatrix(pccluster rc, pccluster cc, uint rsons, uint csons)
    void del_hmatrix(phmatrix hm)

    void clear_hmatrix(hmatrix hm)
    void copy_hmatrix(pchmatrix src, phmatrix trg)
    phmatrix clone_hmatrix(hmatrix hm)
    size_t getsize_hmatrix(hmatrix hm)
    void build_from_block_hmatrix(pcblock b, uint k)
    void norm2_hmatrix(pchmatrix H)



## bem3d.h ##
cdef extern from 'bem3d.h' nogil:

    cdef enum _basisfunctionbem3d:
        BASIS_NONE_BEM3D
        BASIS_CONSTANT_BEM3D
        BASIS_LINEAR_BEM3D
    
    ctypedef _basisfunctionbem3d basisfunctionbem3d

cdef extern from 'bem3d.h' nogil:

    cdef struct _bem3d:
        field k
        field kernel_const
        basisfunctionbem3d row_basis
        basisfunctionbem3d col_basis

    ctypedef _bem3d bem3d
    ctypedef bem3d * pbem3d
    ctypedef const bem3d * pcbem3d

    pbem3d new_bem3d(pcsurface3d gr, basisfunctionbem3d row_basis, basisfunctionbem3d col_basis)
    void del_bem3d(pbem3d bem)

    void setup_hmatrix_aprx_aca_bem3d(pbem3d bem, pccluster rc, pccluster cc, pcblock tree, real accur)
    void setup_hmatrix_aprx_paca_bem3d(pbem3d bem, pccluster rc, pccluster cc, pcblock tree, real accur)
    void setup_hmatrix_aprx_hca_bem3d(pbem3d bem, pccluster rc, pccluster cc, pcblock tree, uint m, real accur)
    void setup_hmatrix_aprx_inter_row_bem3d(pbem3d bem, pccluster rc, pccluster cc, pcblock tree, uint m)
    void assemble_bem3d_hmatrix(pbem3d bem, pblock b, phmatrix G)

    
## helmholtzbem3d.h ##
cdef extern from 'helmholtzbem3d.h' nogil:

    pbem3d new_slp_helmholtz_bem3d(field k, pcsurface3d gr, uint q_regular, uint q_singular, 
        basisfunctionbem3d row_basis, basisfunctionbem3d col_basis)
    pbem3d new_dlp_helmholtz_bem3d(field k, pcsurface3d gr, uint q_regular, uint q_singular, 
        basisfunctionbem3d row_basis, basisfunctionbem3d col_basis, field alpha)


## truncation.h ##
cdef extern from 'truncation.h' nogil:
    
    cdef struct _truncmode:
        bint frobenius
        bint absolute
        bint blocks
        real zeta_level
        real zeta_age
        
    ctypedef _truncmode truncmode
    ctypedef truncmode * ptruncmode
    ctypedef const truncmode * pctruncmode

    ptruncmode new_truncmode()
    ptruncmode new_releucl_truncmode()
    void del_truncmode(ptruncmode tm)


## harith.h ##
cdef extern from 'harith.h' nogil:

    void choldecomp_hmatrix(phmatrix a, pctruncmode tm, real eps)
    void cholsolve_hmatrix_avector(pchmatrix a, pavector x)
    void choleval_hmatrix_avector(pchmatrix a, pavector x)
    void lrdecomp_hmatrix(phmatrix a, pctruncmode tm, real eps)
    void lrsolve_hmatrix_avector(bint atrans, pchmatrix a, pavector x)
    void lreval_n_hmatrix_avector(pchmatrix a, pavector x)


## krylovsolvers.h ##
cdef extern from 'krylovsolvers.h' nogil:

    uint solve_cg_hmatrix_avector(pchmatrix A, pcavector b, pavector x, real eps, uint maxiter)
    uint solve_cg_amatrix_avector(pcamatrix A, pcavector b, pavector x, real eps, uint maxiter)






