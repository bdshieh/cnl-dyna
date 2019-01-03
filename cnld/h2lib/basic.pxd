## basic_cy.pxd ##


from . cimport _basic


ctypedef _basic.real real
ctypedef _basic.field field
ctypedef _basic.uint uint
ctypedef _basic.size_t size_t

from . basic cimport cabs, carg, conj, cexp, creal, cimag
cpdef init_h2lib()
cpdef uninit_h2lib()