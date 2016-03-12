# ########################################################################
# Copyright 2016 Advanced Micro Devices, Inc.
# ########################################################################

# Natively available on including Ubuntu 14.04, OpenSUSE 13.2, CentOS 6.6
cmake_minimum_required( VERSION 2.8.12 )

# if( NOT CMAKE_TOOLCHAIN_FILE )
#   message( FATAL_ERROR "This project expects to know what compilers it should use through CMAKE_TOOLCHAIN_FILE" )
# endif( )

# This project may compile dependencies for clients
project( rocblas-clients CXX )

list( APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake )

include( build-options )

if( BUILD_CLIENTS_DEPENDENCY_BOOST )
  include( external-boost )
endif( )

if( BUILD_CLIENTS_DEPENDENCY_GTEST )
  include( external-gtest )
endif( )

if( BUILD_CLIENTS_DEPENDENCY_LAPACK )
  include( external-lapack )
endif( )

set( CLIENTS_CMAKE_ARGS
	-DCMAKE_TOOLCHAIN_FILE=${DEVICE_TOOLCHAIN_FILE}
	-DBUILD_64=${BUILD_64}
	)

if( NOT CMAKE_CONFIGURATION_TYPES )
	list( APPEND CLIENTS_CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} )
endif( )

if( BUILD_CLIENTS_SAMPLES OR BUILD_CLIENTS_BENCHMARKS OR BUILD_CLIENTS_TESTS )
  include( ExternalProject )
endif( )

# WARNING: do not surround CMAKE_PREFIX_PATH with quotes, it breaks finding modules
# =!!!= -DCMAKE_PREFIX_PATH:PATH="${CMAKE_PREFIX_PATH}"
if( BUILD_CLIENTS_SAMPLES )
  set( SAMPLES_CMAKE_ARGS
        ${CLIENTS_CMAKE_ARGS}
        -DCMAKE_PREFIX_PATH:PATH=${CMAKE_PREFIX_PATH}
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
          -DCMAKE_PREFIX_PATH:PATH=${CMAKE_PREFIX_PATH}^^${BOOST_ROOT}
      )

  ExternalProject_Add( benchmarks
    DEPENDS boost
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
        -DCMAKE_PREFIX_PATH:PATH=${CMAKE_PREFIX_PATH}^^${BOOST_ROOT}^^${GTEST_ROOT}^^${LAPACK_ROOT}
  	)

  ExternalProject_Add( tests
    DEPENDS boost gtest lapack
    SOURCE_DIR ${PROJECT_SOURCE_DIR}/tests
    BINARY_DIR tests-build
    CMAKE_ARGS ${TESTS_CMAKE_ARGS}
    LIST_SEPARATOR ^^
    INSTALL_COMMAND ""
  )
endif( )
