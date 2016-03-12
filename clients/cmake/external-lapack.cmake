# ########################################################################
# Copyright 2016 Advanced Micro Devices, Inc.
# ########################################################################

message( STATUS "Configuring lapack external dependency" )
include( ExternalProject )

set( lapack_git_repository "https://github.com/live-clones/lapack.git" CACHE STRING "URL to download lapack from" )
set( lapack_git_tag "master" CACHE STRING "git branch" )

# Create a workspace to house the src and buildfiles for googleMock
set_directory_properties( PROPERTIES EP_PREFIX ${CMAKE_BINARY_DIR}/extern/lapack )

set( lapack_cmake_args -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>/package -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE} )

# message( STATUS "lapack_make ( " ${lapack_make} " ) " )
# message( STATUS "lapack_cmake_args ( " ${lapack_cmake_args} " ) " )

# Master branch has a new structure that combines googletest with googlemock
ExternalProject_Add(
  lapack
  DOWNLOAD_COMMAND git clone --depth 1 --branch ${lapack_git_tag} ${lapack_git_repository}
  CMAKE_ARGS ${lapack_cmake_args} -DCBLAS=ON -DLAPACKE=ON -DBUILD_TESTING=OFF
)

ExternalProject_Get_Property( lapack source_dir )

set_property( TARGET lapack PROPERTY FOLDER "extern" )
ExternalProject_Get_Property( lapack install_dir )

# For use by the user of Externallapack.cmake
set( LAPACK_ROOT ${install_dir}/package )
