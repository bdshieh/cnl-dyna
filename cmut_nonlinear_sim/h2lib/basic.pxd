## basic.h ##

ctypedef double real
ctypedef double complex field
ctypedef unsigned int uint
ctypedef unsigned long size_t

## complex.h ##
cdef extern from 'complex.h' nogil:

    double cabs(double complex)
    double carg(double complex)
    double complex conj(double complex)
    double complex cexp(double complex)
    double creal(double complex)
    double cimag(double complex)

cdef extern from 'basic.h' nogil:

    void init_h2lib(int *argc, char ***argv)
    void uninit_h2lib()
