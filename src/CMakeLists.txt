cmake_minimum_required(VERSION 2.6.0)
project (cmut-nonlinear-sim)


link_directories(${CMAKE_SOURCE_DIR}/../lib/)
include_directories(${CMAKE_SOURCE_DIR}/../include/)

add_executable (demo demo_zmatrix_hmatrix.c)
target_link_libraries(demo H2LIB winmm)
# add_executable (example example_hmatrix_bem3d.c)
# target_link_libraries(example H2LIB winmm)

add_compile_definitions(USE_COMPLEX)
add_compile_options(-lwinmm)
