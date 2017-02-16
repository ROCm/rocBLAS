# ########################################################################
# Copyright 2016 Advanced Micro Devices, Inc.
# ########################################################################

# Natively available on including Ubuntu 14.04, OpenSUSE 13.2, CentOS 6.6
cmake_minimum_required( VERSION 2.8.12 )

# On windows, it's convenient to change the default install prefix such that it does NOT point to 'program files'
if( WIN32 AND CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT )
  set( CMAKE_INSTALL_PREFIX "${PROJECT_BINARY_DIR}/library-package" CACHE PATH "Install path prefix, prepended onto install directories" FORCE )
endif( )

# This has to be initialized before the project() command appears
# Set the default of CMAKE_BUILD_TYPE to be release, unless user specifies with -D.  MSVC_IDE does not use CMAKE_BUILD_TYPE
if( NOT DEFINED CMAKE_CONFIGURATION_TYPES AND NOT DEFINED CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE "Debug" CACHE STRINGS "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel.")
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS None Debug Release RelWithDebInfo MinSizeRel)
endif()

# Append our library helper cmake path and the cmake path for hip (for convenience)
# Users may override HIP path by specifying their own in CMAKE_MODULE_PATH
list( APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake /opt/rocm/hip/cmake )

# Temp fix, there is a problem building our test program using googletest with gcc
# Change compiler default to clang
if( NOT DEFINED CMAKE_CXX_COMPILER AND NOT DEFINED ENV{CXX} )
  set( CMAKE_CXX_COMPILER clang++ )
endif( )

if( NOT DEFINED CMAKE_C_COMPILER AND NOT DEFINED ENV{CC} )
  set( CMAKE_C_COMPILER clang )
endif( )

# The superbuild does not build anything itself, all compiling is done in external projects
project( rocblas-superbuild NONE )

# Everything is initially off, so that cache is not initialized until user elects to build
option( BUILD_LIBRARY "Build rocBLAS library" OFF )
option( BUILD_CLIENTS "Build rocBLAS clients" OFF )

option( BUILD_WITH_TENSILE "Building rocBLAS with Tensile or not" ON)

# which benchmark solution schedule
set( Tensile_ROOT "" CACHE STRING "Local path of Tensile.")
if ( BUILD_WITH_TENSILE )
  set( Tensile_LOGIC "FijiROCm14" CACHE STRING "Tensile to use which logic?")
  set_property( CACHE Tensile_LOGIC PROPERTY STRINGS
    FijiROCm14
    )
  option( Tensile_MERGE_FILES "Tensile to merge kernels and solutions files?" OFF)
  option( Tensile_SHORT_FILENAMES "Tensile to use short file names? Use if compiler complains they're too long." OFF)
  option( Tensile_PRINT_DEBUG "Tensile to print runtime debug info?" OFF)
endif()

# BUILD_SHARED_LIBS is a cmake built-in; we make it an explicit option such that it shows in cmake-gui
option( BUILD_SHARED_LIBS "Build rocBLAS as a shared library" OFF )

set( HIP_ROOT /opt/rocm/hip CACHE PATH "Specify hip installation dir")
set( BOOST_ROOT /opt/boost  CACHE PATH "Specify boost installation dir")

# set( rocblas_INSTALL_DIR ${CMAKE_INSTALL_PREFIX} )
set( rocblas_INSTALL_DIR "${PROJECT_BINARY_DIR}/package" )

# Default behavior is to NOT install library, but clients may overload
set( rocblas_INSTALL_COMMAND INSTALL_COMMAND ${CMAKE_COMMAND} -E echo_append )

# Clients are programs provided in this repository, that make use of the library as a library client.  This can include
# but is not limited to benchmarks, tests and samples.
if( BUILD_CLIENTS )
  # Clients need to find and link rocblas; we install it locally instead of globally
  unset( rocblas_INSTALL_COMMAND )
endif()

# standard cmake module to manage dependencies
include( ExternalProject )

# This captures all of the dependencies cmake builds itself
set( rocblas_dependencies )

set( BASE_CMAKE_ARGS )

# Noramlize the different ways of specifying a c++ compiler through -DCMAKE_CXX_COMPILER
if( DEFINED CMAKE_CXX_COMPILER )
  message( STATUS "CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER} " )
  list( APPEND BASE_CMAKE_ARGS -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} )
elseif( DEFINED ENV{CXX} )
  message( STATUS "ENV{CXX}: $ENV{CXX} " )
  list( APPEND BASE_CMAKE_ARGS -DCMAKE_CXX_COMPILER=$ENV{CXX} )
endif( )

# Noramlize the different ways of specifying a c compiler through -DCMAKE_C_COMPILER
if( DEFINED CMAKE_C_COMPILER )
  message( STATUS "CMAKE_C_COMPILER: ${CMAKE_C_COMPILER}" )
  list( APPEND BASE_CMAKE_ARGS -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} )
elseif( DEFINED ENV{CC} )
  message( STATUS "ENV{CC}: $ENV{CC} " )
  list( APPEND BASE_CMAKE_ARGS -DCMAKE_C_COMPILER=$ENV{CC} )
endif( )

if( DEFINED CMAKE_CXX_FLAGS )
  message( STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS} " )
  list( APPEND BASE_CMAKE_ARGS -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS} )
endif( )

if( DEFINED CMAKE_C_FLAGS )
  message( STATUS "CMAKE_C_FLAGS: ${CMAKE_C_FLAGS}" )
  list( APPEND BASE_CMAKE_ARGS -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS} )
endif( )

# CMAKE_BUILD_TYPE only applies to single configuration build systems
if( DEFINED CMAKE_BUILD_TYPE )
  list( APPEND BASE_CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} )
endif( )

if( NOT DEFINED DEVICE_CXX_COMPILER )
  find_package( HIP REQUIRED )
  set( DEVICE_CXX_COMPILER ${HIP_ROOT_DIR}/bin/hipcc )
endif()

if( BUILD_LIBRARY )
  if( BUILD_WITH_TENSILE )
    # defines
    # Tensile_INCLUDE_DIRS
    # TensileLib_LIBRARIES
    # TensileLogger_LIBRARIES
    if ( Tensile_ROOT STREQUAL "" )
      include( cmake/external-Tensile.cmake )
      list( APPEND rocblas_dependencies Tensile )
    endif()
    list( APPEND CMAKE_PREFIX_PATH ${Tensile_ROOT} )
  endif()

  # WARNING: do not surround CMAKE_PREFIX_PATH with quotes, it breaks
  # Replace all occurances of ; with ^^, which we elect to use a path seperator
  string(REGEX REPLACE ";" "^^" LIBRARY_PREFIX_PATH "${CMAKE_PREFIX_PATH}" )
  string(REGEX REPLACE ";" "^^" LIBRARY_MODULE_PATH "${CMAKE_MODULE_PATH}" )

  set( LIBRARY_CMAKE_ARGS
    -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
    -DCMAKE_PREFIX_PATH=${LIBRARY_PREFIX_PATH}
    -DCMAKE_MODULE_PATH=${LIBRARY_MODULE_PATH}
    -DBUILD_SHARED_LIBS=${BUILD_SHARED_LIBS}
    -DCMAKE_CXX_COMPILER=${DEVICE_CXX_COMPILER}
    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
    -DBUILD_WITH_TENSILE=${BUILD_WITH_TENSILE}
    -DTensile_LOGIC=${Tensile_LOGIC}
    -DTensile_ROOT=${Tensile_ROOT}
    -DTensile_MERGE_FILES=${Tensile_MERGE_FILES}
    -DTensile_SHORT_FILENAMES=${Tensile_SHORT_FILENAMES}
    -DTensile_PRINT_DEBUG=${Tensile_PRINT_DEBUG}
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
  )

  # Build the library as an external project
  ExternalProject_Add( rocblas
    DEPENDS ${rocblas_dependencies}
    SOURCE_DIR ${PROJECT_SOURCE_DIR}/library
    BINARY_DIR library-build
    INSTALL_DIR library-package
    LIST_SEPARATOR ^^
    CMAKE_ARGS ${LIBRARY_CMAKE_ARGS}
    ${rocblas_INSTALL_COMMAND}
  )
endif( )

# Build clients of the library
if( BUILD_CLIENTS )
  include( clients/cmake/build-options.cmake )

  if( BUILD_LIBRARY )
    ExternalProject_Get_Property( rocblas install_dir )
    list( APPEND CMAKE_PREFIX_PATH ${install_dir} )
  endif( )

  # WARNING: do not surround CMAKE_PREFIX_PATH with quotes, it breaks
  # Replace all occurances of ; with ^^, which we elect to use a path seperator
  string(REGEX REPLACE ";" "^^" CLIENT_PREFIX_PATH "${CMAKE_PREFIX_PATH}" )
  string(REGEX REPLACE ";" "^^" CLIENT_MODULE_PATH "${CMAKE_MODULE_PATH}" )

  # Default arguments that get passed down into all external projects
  set( CLIENTS_CMAKE_ARGS
    -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
    -DDEVICE_CXX_COMPILER=${DEVICE_CXX_COMPILER}
    -DCMAKE_PREFIX_PATH=${CLIENT_PREFIX_PATH}
    -DCMAKE_MODULE_PATH=${CLIENT_MODULE_PATH}
    -DBUILD_CLIENTS_SAMPLES=${BUILD_CLIENTS_SAMPLES}
    -DBUILD_CLIENTS_BENCHMARKS=${BUILD_CLIENTS_BENCHMARKS}
    -DBUILD_CLIENTS_TESTS=${BUILD_CLIENTS_TESTS}
    -DBUILD_WITH_TENSILE=${BUILD_WITH_TENSILE}
    -DDEVICE_CXX_COMPILER=${DEVICE_CXX_COMPILER}
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}

  )

  # Clients are set up as an external project to take advantage of specifying toolchain files.
  # We want cmake to go through it's usual discovery process
  ExternalProject_Add( rocblas-clients
    DEPENDS rocblas
    SOURCE_DIR ${PROJECT_SOURCE_DIR}/clients
    BINARY_DIR clients-build
    INSTALL_DIR clients-package
    LIST_SEPARATOR ^^
    CMAKE_ARGS ${CLIENTS_CMAKE_ARGS}
    INSTALL_COMMAND ""
  )
endif( )
