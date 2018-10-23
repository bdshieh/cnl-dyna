

cdef extern from '<complex.h>' nogil:

    double cabs(double complex)
    double carg(double complex)
    double complex conj(double complex)
    double complex cexp(double complex)
    double creal(double complex)
    double cimag(double complex)


ctypedef double real
ctypedef float complex field
ctypedef unsigned int uint

cdef extern from 'basic.h':
    void init_h2lib(int *argc, char ***argv)
    void uninit_h2lib()


cdef extern from 'avector.h':
    
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


'''
cdef extern from 'amatrix.h':
    pass


cdef extern from 'bem3d.h':
    assemble_bem3d_hmatrix


cdef extern from 'helmholtzbem3d.h':
    new_slp_helmholtz_bem3d
    new_dlp_helmholtz_bem3d

cdef extern from 'hmatrix.h':
    

cdef extern from 'harith.h':
    pass


cdef extern from 'block.h':
    build_nonstrict_block


cdef extern from 'cluster.h':
    pass


cdef extern from 'krylovsolvers.h':
    solve_cg_hmatrix_avector


cdef extern from 'surface3d.h':
    pass


cdef extern from 'macrosurface3d.h':
    pass
'''
