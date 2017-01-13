# ########################################################################
# Copyright 2016 Advanced Micro Devices, Inc.
# ########################################################################

message( STATUS "Configuring lapack external dependency" )
include( ExternalProject )

set( lapack_git_repository "https://github.com/Reference-LAPACK/lapack-release" CACHE STRING "URL to download lapack from" )
set( lapack_git_tag "lapack-3.7.0" CACHE STRING "git branch" )

# If the user does not specify an explicit fortran compiler, assume gfortran
if( NOT DEFINED CMAKE_Fortran_COMPILER )
    set( CMAKE_Fortran_COMPILER gfortran )
endif( )

# If the user does not specify an explicit fortran compiler, assume gfortran
if( NOT DEFINED CMAKE_C_COMPILER )
    set( CMAKE_C_COMPILER cc )
endif( )

set( lapack_cmake_args -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>/package -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} -DCMAKE_Fortran_COMPILER=${CMAKE_Fortran_COMPILER} )

# message( STATUS "lapack_make ( " ${lapack_make} " ) " )
# message( STATUS "lapack_cmake_args ( " ${lapack_cmake_args} " ) " )

include( GNUInstallDirs )

# lapack cmake exports has a bug on debian architectures, they do not take into account the
# lib/<machine> paths
# if CMAKE_INSTALL_LIBDIR is of the form above, strip the machine
# Match against a '/' in CMAKE_INSTALL_LIBDIR, i.e. lib/x86_64-linux-gnu
if( ${CMAKE_INSTALL_LIBDIR} MATCHES "lib/.*" )
  list( APPEND lapack_cmake_args "-DCMAKE_INSTALL_LIBDIR=lib" )
endif( )

# Master branch has a new structure that combines googletest with googlemock
ExternalProject_Add(
  lapack
  PREFIX ${CMAKE_BINARY_DIR}/extern/lapack
  GIT_REPOSITORY ${lapack_git_repository}
  GIT_TAG ${lapack_git_tag}
  CMAKE_ARGS ${lapack_cmake_args} -DCBLAS=ON -DLAPACKE=OFF -DBUILD_TESTING=OFF
  LOG_BUILD 1
  LOG_INSTALL 1
)

ExternalProject_Get_Property( lapack source_dir )

set_property( TARGET lapack PROPERTY FOLDER "extern" )
ExternalProject_Get_Property( lapack install_dir )

# For use by the user of Externallapack.cmake
set( LAPACK_ROOT ${install_dir}/package )
