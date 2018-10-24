

cdef extern from 'stddef.h' nogil:

    cdef size_t


cdef extern from '<complex.h>' nogil:

    double cabs(double complex)
    double carg(double complex)
    double complex conj(double complex)
    double complex cexp(double complex)
    double creal(double complex)


ctypedef double real
ctypedef float complex field
ctypedef unsigned int uint


cdef extern from 'basic.h' nogil:
    void init_h2lib(int *argc, char ***argv)
    void uninit_h2lib()


cdef extern from 'avector.h' nogil:
    
    ctypedef struct _avector
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


cdef extern from 'amatrix.h' nogil:

    ctypedef struct _amatrix
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
    void copy_amatrix(bool atrans, pcamatrix a, pamatrix b)
    pamatrix clone_amatrix(pcamatrix src)
    void scale_amatrix(field alpha, pamatrix a)
    void conjugate_amatrix(pamatrix a)
    real norm2_amatrix(pcamatrix a)
    real norm2diff_amatrix(pcamatrix a, pcamatrix b)
    void addeval_amatrix_avector(field alpha, pcamatrix a, pcavector, src, pavector trg
    void add_amatrix(field alpha, bool atrans, pcamatrix a, pamatrix b)


cdef extern from 'bem3d.h' nogil:

    ctypedef struct _bem3d
    ctypedef _bem3d bem3d
    ctypedef bem3d * pbem3d
    ctypedef const bem3d * pcbem3d

    void setup_hmatrix_aprx_aca_bem3d(pbem3d bem, pccluster rc, pccluster cc, pcblock tree, real accur)
    void setup_hmatrix_aprx_paca_bem3d(pbem3d bem, pccluster rc, pccluster cc, pcblock tree, real accur)
    void setup_hmatrix_aprx_hca_bem3d(pbem3d bem, pccluster rc, pccluster cc, pcblock tree, uint m, real accur)
    void setup_hmatrix_aprx_inter_row_bem3d(pbem3d bem, pccluster rc, pccluster cc, pcblock tree, uint m)
    void assemble_bem3d_hmatrix(pbem3d bem, pblock b, phmatrix G)


cdef extern from 'helmholtzbem3d.h' nogil:
    new_slp_helmholtz_bem3d
    new_dlp_helmholtz_bem3d


cdef extern from 'hmatrix.h' nogil:

    ctypedef struct _hmatrix
    ctypedef _hmatrix hmatrix
    ctypedef hmatrix * phmatrix
    ctypedef const hmatrix * pchmatrix

    phmatrix new_hmatrix(pccluster rc, pccluster cc)
    phmatrix new_super_hmatrix(pccluster rc, pccluster cc, uint rsons, uint csons)
    void del_hmatrix(hmatrix hm)
    void clear_hmatrix(hmatrix hm)
    void copy_hmatrix(pchmatrix src, phmatrix trg)
    phmatrix clone_hmatrix(hmatrix hm)
    size_t getsize_hmatrix(hmatrix hm)
    void build_from_block_hmatrix(pcblock b, uint k)
    void norm2_hmatrix(pchmatrix H)


cdef extern from 'harith.h' nogil:

    void choldecomp_hmatrix(phmatrix a, pctruncmode tm, real eps)
    void cholsolve_hmatrix_avector(pchmatrix a, pavector x)
    void choleval_hmatrix_avector(pchmatrix a, pavector x)
    void lrdecomp_hmatrix(phmatrix a, pctruncmode tm, real eps)
    void lrsolve_hmatrix_avector(bool atrans, pchmatrix a, pavector x)
    void lreval_n_hmatrix_avector(pchmatrix a, pavector x)


cdef extern from 'truncation.h' nogil:
    
    ctypedef struct _truncmode
    ctypedef _truncmode truncmode
    ctypedef truncmode * ptruncmode
    ctypedef const truncmode * pctruncmode


cdef extern from 'block.h' nogil:

    ctypedef struct _block
    ctypedef _block block
    ctypedef block * pblock
    ctypedef const block * pcblock
    ctypedef bool (* admissible) (pcluster rc, pcluster cc, void * data)

    pblock build_nonstrict_block(pcluster rc, pcluster cc, void * data, admissible admis)


cdef extern from 'cluster.h' nogil:
    
    ctypedef struct _cluster
    ctypedef _cluster cluster
    ctypedef cluster * pcluster
    ctypedef const cluster * pccluster


cdef extern from 'krylovsolvers.h' nogil:
    solve_cg_hmatrix_avector


cdef extern from 'surface3d.h' nogil:
    pass


cdef extern from 'macrosurface3d.h' nogil:
    new_macrosurface3d
    del_macrosurface3d

