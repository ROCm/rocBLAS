# ########################################################################
# Copyright 2016 Advanced Micro Devices, Inc.
# ########################################################################

# Natively available on including Ubuntu 14.04, OpenSUSE 13.2, CentOS 6.6
cmake_minimum_required( VERSION 2.8.12 )

if( CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT )
	message( FATAL_ERROR "The clients project expects CMAKE_INSTALL_PREFIX to be defined on execution" )
endif( )

# This has to be initialized before the project() command appears
if( NOT CMAKE_BUILD_TYPE )
	message( FATAL_ERROR "The clients project expects CMAKE_BUILD_TYPE to be defined on execution" )
endif()

# The clients project does not build anything itself, but launches independent
# external projects that are clients of the library
project( rocblas-clients CXX )

list( APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake )

include( build-options )

if( BUILD_CLIENTS_DEPENDENCY_BOOST )
	include( external-boost )
endif( )

if( BUILD_CLIENTS_DEPENDENCY_GTEST )
	include( external-gtest )
endif( )

if( BUILD_BENCHMARKS )
	ExternalProject_Add( benchmarks
		DEPENDS boost
		SOURCE_DIR ${PROJECT_SOURCE_DIR}/benchmarks
		BINARY_DIR benchmarks-build
		CMAKE_ARGS ...
		INSTALL_COMMAND ""
	)
endif( )

if( BUILD_TESTS )
	ExternalProject_Add( tests
		DEPENDS boost
		SOURCE_DIR ${PROJECT_SOURCE_DIR}/tests
		BINARY_DIR tests-build
		CMAKE_ARGS ...
		INSTALL_COMMAND ""
	)
endif( )

if( BUILD_SAMPLES )
	ExternalProject_Add( samples
		DEPENDS boost
		SOURCE_DIR ${PROJECT_SOURCE_DIR}/samples
		BINARY_DIR samples-build
		CMAKE_ARGS ...
		INSTALL_COMMAND ""
	)
endif( )
