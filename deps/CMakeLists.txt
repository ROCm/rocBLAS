# ########################################################################
# Copyright 2016-2020 Advanced Micro Devices, Inc.
# ########################################################################
# Helper cmake script to automate building dependencies for rocblas
# This script can be invoked manually by the user with 'cmake -P'

# The ROCm platform requires Ubuntu 16.04 or Fedora 24, which has cmake 3.5
cmake_minimum_required( VERSION 3.5 )

list( APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../cmake )

# This has to be initialized before the project() command appears
# Set the default of CMAKE_BUILD_TYPE to be release, unless user specifies with -D.  MSVC_IDE does not use CMAKE_BUILD_TYPE
if( NOT DEFINED CMAKE_CONFIGURATION_TYPES AND NOT DEFINED CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." )
endif()

# The superbuild does not build anything itself; all compiling is done in external projects
project( rocblas-dependencies NONE )

option( BUILD_BOOST "Download and build boost library" ON )
option( BUILD_GTEST "Download and build googletest library" ON )
option( BUILD_LAPACK "Download and build lapack library" ON )
# option( BUILD_VERBOSE "Print helpful build debug information" OFF )

# if( BUILD_VERBOSE )
#   message( STATUS "CMAKE_MODULE_PATH: ${CMAKE_MODULE_PATH}" )
#   message( STATUS "CMAKE_BINARY_DIR: ${CMAKE_BINARY_DIR}" )
#   message( STATUS "CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}" )
#   message( STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}" )
#   message( STATUS "CMAKE_CURRENT_BINARY_DIR: ${CMAKE_CURRENT_BINARY_DIR}" )
#   message( STATUS "CMAKE_CURRENT_LIST_DIR: ${CMAKE_CURRENT_LIST_DIR}" )
#   message( STATUS "CMAKE_CURRENT_LIST_FILE: ${CMAKE_CURRENT_LIST_FILE}" )
# endif( )

# This module scrapes the CMakeCache.txt file and attempts to get all the cli options the user specified to cmake invocation
include( get-cli-arguments )

# The following is a series of super-build projects; this cmake project will download and build
if( BUILD_GTEST )
  include( external-gtest )

  list( APPEND rocblas_dependencies googletest )
  set( gtest_custom_target COMMAND cd ${GTEST_BINARY_ROOT}$<SEMICOLON> ${CMAKE_COMMAND} --build . --target install )
endif( )

if( BUILD_LAPACK )
  include( external-lapack )

  list( APPEND rocblas_dependencies lapack )
  set( lapack_custom_target COMMAND cd ${LAPACK_BINARY_ROOT}$<SEMICOLON> ${CMAKE_COMMAND} --build . --target install )
endif( )

if( BUILD_BOOST )
  include( external-boost )

  list( APPEND rocblas_dependencies boost )
  set( boost_custom_target COMMAND cd ${BOOST_BINARY_ROOT}$<SEMICOLON> ${Boost.Command} install )
endif( )

# POLICY CMP0037 - "Target names should not be reserved and should match a validity pattern"
# Familiar target names like 'install' should be OK at the super-build level
if( POLICY CMP0037 )
  cmake_policy( SET CMP0037 OLD )
endif( )

add_custom_target( install
  ${boost_custom_target}
  ${gtest_custom_target}
  ${lapack_custom_target}
  DEPENDS ${rocblas_dependencies}
)
