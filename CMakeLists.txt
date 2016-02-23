# ########################################################################
# Copyright 2016 Advanced Micro Devices, Inc.
# ########################################################################

# Natively available on including Ubuntu 14.04, OpenSUSE 13.2, CentOS 6.6
cmake_minimum_required( VERSION 2.8.12 )

# On windows, it's convenient to change the default install prefix such that it does NOT point to 'program files'
if( WIN32 AND CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT )
	set( CMAKE_INSTALL_PREFIX "${PROJECT_BINARY_DIR}/rocblas-package" CACHE PATH "Install path prefix, prepended onto install directories" FORCE )
endif( )

# This has to be initialized before the project() command appears
# Set the default of CMAKE_BUILD_TYPE to be release, unless user specifies with -D.  MSVC_IDE does not use CMAKE_BUILD_TYPE
if( NOT MSVC_IDE AND NOT CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." FORCE )
endif()

# Default arguments that get passed down into all external projects
set( rocblas_CMAKE_ARGS
  -DCMAKE_TOOLCHAIN_FILE=${PROJECT_SOURCE_DIR}/cmake/hcc-toolchain.cmake
  -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
  -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
  )

# Everything is initially off, so that cache is not initialized until user elects to build
option( BUILD_LIBRARY "Build rocBLAS library" OFF )
option( BUILD_CLIENTS "Build rocBLAS clients" OFF )

# set( rocblas_INSTALL_DIR ${CMAKE_INSTALL_PREFIX} )
set( rocblas_INSTALL_DIR "${PROJECT_BINARY_DIR}/rocblas-package" )

# Default behavior is to NOT install
set( rocblas_INSTALL_COMMAND INSTALL_COMMAND "." )

# Clients are programs provided in this repository, that make use of the library as a library client.  This can include
# but is not limited to benchmarks, tests and samples.
if( BUILD_CLIENTS )
  include( clients/cmake/build-options.cmake )

  # Clients need to find and link rocblas; we install it locally instead of globally
  set( rocblas_INSTALL_DIR "${PROJECT_BINARY_DIR}/rocblas-package" )
  unset( rocblas_INSTALL_COMMAND )
endif()

# standard cmake module to manage dependencies
include( ExternalProject )

if( BUILD_LIBRARY )
  include( src/cmake/build-options.cmake )

  # Build the library as an external project
  ExternalProject_Add( rocblas
    SOURCE_DIR ${PROJECT_SOURCE_DIR}/src
    BINARY_DIR rocblas-build
    INSTALL_DIR ${rocblas_INSTALL_DIR}
    CMAKE_ARGS ${rocblas_CMAKE_ARGS} -DBUILD_LIBRARY_64=${BUILD_LIBRARY_64} -DBUILD_LIBRARY_UNICODE=${BUILD_LIBRARY_UNICODE}
    ${rocblas_INSTALL_COMMAND}
  )
endif()

# Build clients of the library
# if( BUILD_TESTS OR BUILD_BENCHMARKS OR BUILD_SAMPLES )
#   add_subdirectory( clients )
# endif( )
