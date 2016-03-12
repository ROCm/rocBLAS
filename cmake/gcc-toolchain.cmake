# Trivial toolchain file to help project pick up appropriate compilers
set( CMAKE_C_COMPILER gcc-4.8 )
set( CMAKE_CXX_COMPILER g++-4.8 )

# we use gfortran to resolve fortran dependencies for lapack
set( CMAKE_Fortran_COMPILER gfortran-4.8 )
