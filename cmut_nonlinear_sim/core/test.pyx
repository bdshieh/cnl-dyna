
from . libh2 cimport *


cdef real a = 5
print(a)

init_h2lib(NULL, NULL)

# cdef pavector vec = new_avector(5)
# del_avector(vec)

uninit_h2lib()
