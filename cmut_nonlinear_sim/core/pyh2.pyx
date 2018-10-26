
from . libh2 cimport *
# from . cimport libh2


cdef real a = 5
print(a)

init_h2lib(NULL, NULL)


cdef pavector vec = new_avector(5)
print(vec.v[0])

del_avector(vec)
print('vector')

cdef pamatrix mat = new_amatrix(10, 10)
del_amatrix(mat)
print('matrix')



uninit_h2lib()
