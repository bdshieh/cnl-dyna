## basic_cy.pyx ##


from . cimport _basic


cpdef init_h2lib():
    _basic.init_h2lib(NULL, NULL)

cpdef uninit_h2lib():
    _basic.uninit_h2lib()
