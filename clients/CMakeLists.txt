# ########################################################################
# Copyright 2016 Advanced Micro Devices, Inc.
# ########################################################################

# Natively available on including Ubuntu 14.04, OpenSUSE 13.2, CentOS 6.6
cmake_minimum_required( VERSION 2.8.12 )

if( NOT CMAKE_TOOLCHAIN_FILE )
  message( FATAL_ERROR "This project expects to know what compilers it should use through CMAKE_TOOLCHAIN_FILE" )
endif( )

if( NOT CMAKE_PREFIX_PATH )
  message( FATAL_ERROR "This project expects to know what where to find the library through CMAKE_PREFIX_PATH" )
endif( )

# The clients project does not build anything itself, but launches independent
# external projects that are clients of the library
project( rocblas-clients CXX )

list( APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake )

include( build-options )

set( clients_CMAKE_ARGS
	-DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}
	-DCMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}
	-DBUILD_64=${BUILD_64}
	)

if( NOT CMAKE_CONFIGURATION_TYPES )
	list( APPEND clients_CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} )
endif( )

if( BUILD_CLIENTS_DEPENDENCY_BOOST )
  include( external-boost )
endif( )

if( BUILD_CLIENTS_DEPENDENCY_GTEST )
  include( external-gtest )
endif( )

include( ExternalProject )

if( BUILD_CLIENTS_BENCHMARKS )
  ExternalProject_Add( benchmarks
    DEPENDS boost
    SOURCE_DIR ${PROJECT_SOURCE_DIR}/benchmarks
    BINARY_DIR benchmarks-build
    CMAKE_ARGS
    INSTALL_COMMAND ""
  )
endif( )

if( BUILD_CLIENTS_TESTS )
  ExternalProject_Add( tests
    DEPENDS gtest
    SOURCE_DIR ${PROJECT_SOURCE_DIR}/tests
    BINARY_DIR tests-build
    CMAKE_ARGS
    INSTALL_COMMAND ""
  )
endif( )

if( BUILD_CLIENTS_SAMPLES )
  ExternalProject_Add( samples
    SOURCE_DIR ${PROJECT_SOURCE_DIR}/samples
    BINARY_DIR samples-build
    CMAKE_ARGS ${clients_CMAKE_ARGS}
    INSTALL_COMMAND ""
  )
endif( )
