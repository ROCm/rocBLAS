# ########################################################################
# Copyright 2016 Advanced Micro Devices, Inc.
# ########################################################################

message( STATUS "Configuring lapack external dependency" )
include( ExternalProject )

set( lapack_git_repository "https://github.com/live-clones/lapack.git" CACHE STRING "URL to download lapack from" )
set( lapack_git_tag "master" CACHE STRING "git branch" )

set( lapack_cmake_args -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>/package -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE} )

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
  DOWNLOAD_COMMAND git clone --depth 1 --branch ${lapack_git_tag} ${lapack_git_repository}
  CMAKE_ARGS ${lapack_cmake_args} -DCBLAS=ON -DLAPACKE=OFF -DBUILD_TESTING=OFF
  LOG_BUILD 1
  LOG_INSTALL 1
)

ExternalProject_Get_Property( lapack source_dir )

set_property( TARGET lapack PROPERTY FOLDER "extern" )
ExternalProject_Get_Property( lapack install_dir )

# For use by the user of Externallapack.cmake
set( LAPACK_ROOT ${install_dir}/package )
