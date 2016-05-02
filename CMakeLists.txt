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
if( NOT CMAKE_CONFIGURATION_TYPES )
  set( CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." FORCE )
endif()

# The superbuild does not build anything itself, all compiling is done in external projects
project( rocblas-superbuild NONE )

# Everything is initially off, so that cache is not initialized until user elects to build
option( BUILD_LIBRARY "Build rocBLAS library" OFF )
option( BUILD_CLIENTS "Build rocBLAS clients" OFF )

# Determine whether to build 64-bit (default) or 32-bit
if( MSVC_IDE )
	set( BUILD_64 ${CMAKE_CL_64} )
else()
	option( BUILD_64 "Build a 64-bit product" ON )
endif()

set( HOST_TOOLCHAIN_NAME "clang" CACHE STRING "Compiler toolchain: (clang gcc)" )
set_property( CACHE HOST_TOOLCHAIN_NAME PROPERTY STRINGS clang gcc )

set( DEVICE_TOOLCHAIN_NAME "hipcc" CACHE STRING "Compiler toolchain for library: (hcc hipcc)" )
set_property( CACHE DEVICE_TOOLCHAIN_NAME PROPERTY STRINGS hcc hipcc )

# set( rocblas_INSTALL_DIR ${CMAKE_INSTALL_PREFIX} )
set( rocblas_INSTALL_DIR "${PROJECT_BINARY_DIR}/rocblas-package" )

# Default behavior is to NOT install library, but clients may overload
set( rocblas_INSTALL_COMMAND INSTALL_COMMAND ${CMAKE_COMMAND} -E echo_append )

# Clients are programs provided in this repository, that make use of the library as a library client.  This can include
# but is not limited to benchmarks, tests and samples.
if( BUILD_CLIENTS )
  # Clients need to find and link rocblas; we install it locally instead of globally
  set( rocblas_INSTALL_DIR "${PROJECT_BINARY_DIR}/rocblas-package" )
  unset( rocblas_INSTALL_COMMAND )
endif()

# standard cmake module to manage dependencies
include( ExternalProject )

set( DEVICE_TOOLCHAIN_FILE "" )
# If our toolchain is using hip, we need to download hip and generate a toolchain file so we can find it
# since the tools are not in the standard system path
if( DEVICE_TOOLCHAIN_NAME STREQUAL "hipcc" )
	include( cmake/external-hip.cmake )

	# configure toolchain file to find the hip compiler with the hip we just downloaded
	configure_file( "${PROJECT_SOURCE_DIR}/cmake/${DEVICE_TOOLCHAIN_NAME}-toolchain.cmake.in"
			"${PROJECT_BINARY_DIR}/cmake/${DEVICE_TOOLCHAIN_NAME}-toolchain.cmake" )

	set( DEVICE_TOOLCHAIN_FILE "${PROJECT_BINARY_DIR}/cmake/${DEVICE_TOOLCHAIN_NAME}-toolchain.cmake" )
else( )
	set( DEVICE_TOOLCHAIN_FILE "${PROJECT_SOURCE_DIR}/cmake/${DEVICE_TOOLCHAIN_NAME}-toolchain.cmake" )
endif( )

# Default arguments that get passed down into all external projects
set( BASE_CMAKE_ARGS
	-DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
	-DBUILD_64=${BUILD_64} )

# CMAKE_BUILD_TYPE only applies to single configuration build systems
if( NOT CMAKE_CONFIGURATION_TYPES )
	list( APPEND BASE_CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} )
endif( )

if( BUILD_LIBRARY )
  include( src/cmake/build-options.cmake )

	set( LIBRARY_CMAKE_ARGS
	 ${BASE_CMAKE_ARGS}
	-DCMAKE_TOOLCHAIN_FILE=${DEVICE_TOOLCHAIN_FILE}
	-DBUILD_LIBRARY_UNICODE=${BUILD_LIBRARY_UNICODE} )

  # Build the library as an external project
  ExternalProject_Add( rocblas
    SOURCE_DIR ${PROJECT_SOURCE_DIR}/src
    BINARY_DIR rocblas-build
    INSTALL_DIR ${rocblas_INSTALL_DIR}
    CMAKE_ARGS ${LIBRARY_CMAKE_ARGS}
    ${rocblas_INSTALL_COMMAND}
  )
endif()

# Build clients of the library
if( BUILD_CLIENTS )
	include( clients/cmake/build-options.cmake )

	# Default arguments that get passed down into all external projects
	set( CLIENTS_CMAKE_ARGS
	  ${BASE_CMAKE_ARGS}
		-DBUILD_CLIENTS_DEPENDENCY_BOOST=${BUILD_CLIENTS_DEPENDENCY_BOOST}
		-DBUILD_CLIENTS_DEPENDENCY_GTEST=${BUILD_CLIENTS_DEPENDENCY_GTEST}
		-DBUILD_CLIENTS_DEPENDENCY_LAPACK=${BUILD_CLIENTS_DEPENDENCY_LAPACK}
		-DBUILD_CLIENTS_SAMPLES=${BUILD_CLIENTS_SAMPLES}
		-DBUILD_CLIENTS_BENCHMARKS=${BUILD_CLIENTS_BENCHMARKS}
		-DBUILD_CLIENTS_TESTS=${BUILD_CLIENTS_TESTS}
		-DCMAKE_TOOLCHAIN_FILE=${PROJECT_SOURCE_DIR}/cmake/${HOST_TOOLCHAIN_NAME}-toolchain.cmake
		-DDEVICE_TOOLCHAIN_FILE=${DEVICE_TOOLCHAIN_FILE}
	  )

		if( BUILD_LIBRARY )
			ExternalProject_Get_Property( rocblas install_dir )
			list( APPEND CLIENTS_CMAKE_ARGS -DCMAKE_PREFIX_PATH=${install_dir} )
		endif( )

	# Clients are set up as an external project to take advantage of specifying toolchain files.
	# We want cmake to go through it's usual discovery process
  ExternalProject_Add( rocblas-clients
    SOURCE_DIR ${PROJECT_SOURCE_DIR}/clients
    BINARY_DIR rocblas-clients-build
    INSTALL_DIR rocblas-clients-package
    CMAKE_ARGS ${CLIENTS_CMAKE_ARGS}
		INSTALL_COMMAND ""
  )
endif( )
