# ########################################################################
# Copyright 2016 Advanced Micro Devices, Inc.
# ########################################################################
# Helper cmake script to automate building dependencies for rocblas
# This script can be invoked manually by the user with 'cmake -P'

# The ROCm platform requires Ubuntu 16.04 or Fedora 24, which has cmake 3.5
cmake_minimum_required( VERSION 3.5 )

# On windows, it's convenient to change the default install prefix such that it does NOT point to 'program files'
if( CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT )
  message( AUTHOR_WARNING "Setting install root to: ${PROJECT_BINARY_DIR}/packages for convenience" )
  message( AUTHOR_WARNING "If another install path is desired, pass -DCMAKE_INSTALL_PREFIX=<> to configure line " )
  set( CMAKE_INSTALL_PREFIX "${PROJECT_BINARY_DIR}/packages" CACHE PATH "Install path prefix, prepended onto install directories" FORCE )
endif( )

# This has to be initialized before the project() command
# Set the default of CMAKE_BUILD_TYPE to be release, unless user specifies with -D.  MSVC_IDE does not use CMAKE_BUILD_TYPE
if( NOT DEFINED CMAKE_CONFIGURATION_TYPES AND NOT DEFINED CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." FORCE )
endif()

list( APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../cmake )

# The superbuild does not build anything itself, all compiling is done in external projects
project( rocblas-dependencies NONE )
include( build-bitness )

option( BUILD_BOOST "Download and build boost library" OFF )
option( BUILD_GTEST "Download and build boost library" OFF )
option( BUILD_LAPACK "Download and build boost library" OFF )
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

if( BUILD_BOOST )
  set( PREFIX_BOOST ${CMAKE_INSTALL_PREFIX} CACHE PATH "Location where boost should install, defaults to /usr/local" )
  include( external-boost )

  list( APPEND rocblas_dependencies boost )
  set( boost_custom_target COMMAND cd ${BOOST_BINARY_ROOT}$<SEMICOLON> ${CMAKE_COMMAND} --build . --target install )
endif( )

if( BUILD_GTEST )
  set( PREFIX_GTEST ${CMAKE_INSTALL_PREFIX} CACHE PATH "Location where boost should install, defaults to /usr/local" )
  include( external-gtest )

  list( APPEND rocblas_dependencies googletest )
  set( gtest_custom_target COMMAND cd ${GTEST_BINARY_ROOT}$<SEMICOLON> ${CMAKE_COMMAND} --build . --target install )
endif( )

if( BUILD_LAPACK )
  set( PREFIX_LAPACK ${CMAKE_INSTALL_PREFIX} CACHE PATH "Location where lapack should install, defaults to /usr/local" )
  include( external-lapack )

  list( APPEND rocblas_dependencies lapack )
  set( lapack_custom_target COMMAND cd ${LAPACK_BINARY_ROOT}$<SEMICOLON> ${CMAKE_COMMAND} --build . --target install )
endif( )

# POLICY CMP0037 - "Target names should not be reserved and should match a validity pattern"
# target names like 'install' should be OK at the superbuild level
if( POLICY CMP0037 )
  cmake_policy( SET CMP0037 OLD )
endif( )

#   COMMAND cd ${GTEST_BINARY_ROOT}$<SEMICOLON> ${CMAKE_COMMAND} --build . --target install

add_custom_target( install
  ${boost_custom_target}
  ${gtest_custom_target}
  ${lapack_custom_target}
  DEPENDS ${rocblas_dependencies}
)
