# Trivial toolchain file to help project pick up appropriate compilers
set( CMAKE_C_COMPILER gcc )
set( CMAKE_CXX_COMPILER g++ )

# we use gfortran to resolve fortran dependencies for lapack
set( CMAKE_Fortran_COMPILER gfortran )
