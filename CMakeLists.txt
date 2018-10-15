cmake_minimum_required(VERSION 2.6.0)
project (cmut-nonlinear-sim)


# add_library (H2LIB STATIC "C:/Home/Projects/H2Lib/Build/Library/Release/H2LIB.lib")
include_directories ("C:/Home/H2Lib/include")

add_executable (demo demo_zmatrix_hmatrix.c)
target_link_libraries (demo "C:/Home/H2Lib/lib/libH2LIB.a")

add_compile_definitions(USE_COMPLEX)