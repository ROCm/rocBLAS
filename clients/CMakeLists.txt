# ########################################################################
# Copyright 2016 Advanced Micro Devices, Inc.
# ########################################################################

# The ROCm platform requires Ubuntu 16.04 or Fedora 24, which has cmake 3.5
cmake_minimum_required( VERSION 3.5 )

# This project may compile dependencies for clients
project( rocblas-clients CXX )

list( APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake )

include( build-options )
include( build-bitness )

# This captures all of the dependencies cmake builds itself
set( rocblas_clients_dependencies )

# If the user does not explicitely specify BOOST_ROOT, build our
# own on the fly
# disable the download, as it often fails
#if( BUILD_CLIENTS_BENCHMARKS )
#  if( NOT DEFINED BOOST_ROOT )
#    include( external-boost )
#    list( APPEND rocblas_clients_dependencies boost )
#  endif( )
#endif()

# if( BUILD_CLIENTS_TESTS )
#   if( NOT DEFINED GTEST_ROOT )
#     include( external-gtest )
#     list( APPEND rocblas_clients_dependencies googletest )
#   endif( )
# endif()

if( BUILD_CLIENTS_BENCHMARKS OR BUILD_CLIENTS_TESTS )
  if( NOT DEFINED LAPACK_ROOT )
    include( external-lapack )
    list( APPEND rocblas_clients_dependencies lapack )
  endif( )
endif()

# WARNING: do not surround CMAKE_PREFIX_PATH with quotes, it breaks
# Replace all occurances of ; with ^^, which we elect to use a path seperator
string(REGEX REPLACE ";" "^^" CLIENT_PREFIX_PATH "${CMAKE_PREFIX_PATH}" )
string(REGEX REPLACE ";" "^^" CLIENT_MODULE_PATH "${CMAKE_MODULE_PATH}" )

set( CLIENTS_CMAKE_ARGS
  -DCMAKE_PREFIX_PATH=${CLIENT_PREFIX_PATH}
  -DCMAKE_MODULE_PATH=${CMAKE_MODULE_PATH}
  -DBUILD_64=${BUILD_64}
  )

# Linking rocblas dynamically should allow the user to use any host compiler
# Linking rocblas statically requires the user to use the hcc compiler
if( BUILD_SHARED_LIBS )
  # Pass through compiler if explicitely set, otherwise let default handling
  if( DEFINED CMAKE_CXX_COMPILER )
    list( APPEND CLIENTS_CMAKE_ARGS -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} )
  endif()
else()
  if( DEFINED DEVICE_CXX_COMPILER )
    list( APPEND CLIENTS_CMAKE_ARGS -DCMAKE_CXX_COMPILER=${DEVICE_CXX_COMPILER} )
  else()
    message( FATAL_ERROR "Statically linking rocblas requires DEVICE_CXX_COMPILER to be defined, typically hipcc")
  endif()
endif( )

if( DEFINED CMAKE_C_COMPILER )
  list( APPEND CLIENTS_CMAKE_ARGS -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} )
endif( )

if( DEFINED CMAKE_CXX_FLAGS )
  list( APPEND CLIENTS_CMAKE_ARGS -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS} )
endif( )

if( DEFINED CMAKE_C_FLAGS )
  list( APPEND CLIENTS_CMAKE_ARGS -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS} )
endif( )

if( NOT DEFINED CMAKE_CONFIGURATION_TYPES AND DEFINED CMAKE_BUILD_TYPE )
  list( APPEND CLIENTS_CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} )
endif( )

if( BUILD_CLIENTS_SAMPLES OR BUILD_CLIENTS_BENCHMARKS OR BUILD_CLIENTS_TESTS )
  include( ExternalProject )
endif( )

if( BUILD_CLIENTS_SAMPLES )
  set( SAMPLES_CMAKE_ARGS
        ${CLIENTS_CMAKE_ARGS}
        -DCMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}
    )

  ExternalProject_Add( samples
    SOURCE_DIR ${PROJECT_SOURCE_DIR}/samples
    BINARY_DIR samples-build
    CMAKE_ARGS ${SAMPLES_CMAKE_ARGS}
    LIST_SEPARATOR ^^
    INSTALL_COMMAND ""
  )
endif( )

if( BUILD_CLIENTS_BENCHMARKS )
    set( BENCH_CMAKE_ARGS
          ${CLIENTS_CMAKE_ARGS}
          -DCMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}^^${BOOST_ROOT}
          -DBUILD_WITH_TENSILE=${BUILD_WITH_TENSILE}
      )

  ExternalProject_Add( benchmarks
    DEPENDS ${rocblas_clients_dependencies}
    SOURCE_DIR ${PROJECT_SOURCE_DIR}/benchmarks
    BINARY_DIR benchmarks-build
    CMAKE_ARGS ${BENCH_CMAKE_ARGS}
    LIST_SEPARATOR ^^
    INSTALL_COMMAND ""
  )
endif( )

if( BUILD_CLIENTS_TESTS )
  set( TESTS_CMAKE_ARGS
        ${CLIENTS_CMAKE_ARGS}
        -DCMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}^^${BOOST_ROOT}^^${GTEST_ROOT}^^${LAPACK_ROOT}
        -DBUILD_WITH_TENSILE=${BUILD_WITH_TENSILE}
    )

  ExternalProject_Add( tests
    DEPENDS ${rocblas_clients_dependencies}
    SOURCE_DIR ${PROJECT_SOURCE_DIR}/gtest
    BINARY_DIR tests-build
    CMAKE_ARGS ${TESTS_CMAKE_ARGS}
    LIST_SEPARATOR ^^
    INSTALL_COMMAND ""
  )
endif( )
