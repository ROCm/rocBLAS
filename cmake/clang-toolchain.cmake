# Trivial toolchain file to help project pick up appropriate compilers
set( CMAKE_C_COMPILER clang-3.5 )
set( CMAKE_CXX_COMPILER clang++-3.5 )

# Clang does not provide a fortran compiler;
# we use gfortran to resolve fortran dependencies for lapack
set( CMAKE_Fortran_COMPILER gfortran-4.8 )
