# ########################################################################
# Copyright 2016 Advanced Micro Devices, Inc.
# ########################################################################

# The ROCm platform requires Ubuntu 16.04 or Fedora 24, which has cmake 3.5
cmake_minimum_required( VERSION 3.5 )

# Consider removing this in the future
# This should appear before the project command, because it does not use FORCE
if( WIN32 )
  set( CMAKE_INSTALL_PREFIX "${PROJECT_BINARY_DIR}/package" CACHE PATH "Install path prefix, prepended onto install directories" )
else( )
  set( CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH "Install path prefix, prepended onto install directories" )
endif( )

# This has to be initialized before the project() command appears
# Set the default of CMAKE_BUILD_TYPE to be release, unless user specifies with -D.  MSVC_IDE does not use CMAKE_BUILD_TYPE
if( NOT DEFINED CMAKE_CONFIGURATION_TYPES AND NOT DEFINED CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." )
endif()

project( rocblas LANGUAGES CXX )

# This finds the rocm-cmake project, and installs it if not found
# rocm-cmake contains common cmake code for rocm projects to help setup and install
set( PROJECT_EXTERN_DIR ${CMAKE_CURRENT_BINARY_DIR}/extern )
find_package( ROCM CONFIG QUIET PATHS /opt/rocm )
if( NOT ROCM_FOUND )
  set( rocm_cmake_tag "master" CACHE STRING "rocm-cmake tag to download" )
  file( DOWNLOAD https://github.com/RadeonOpenCompute/rocm-cmake/archive/${rocm_cmake_tag}.zip
    ${PROJECT_EXTERN_DIR}/rocm-cmake-${rocm_cmake_tag}.zip )

  execute_process( COMMAND ${CMAKE_COMMAND} -E tar xzf ${PROJECT_EXTERN_DIR}/rocm-cmake-${rocm_cmake_tag}.zip
    WORKING_DIRECTORY ${PROJECT_EXTERN_DIR} )

  find_package( ROCM REQUIRED CONFIG PATHS ${PROJECT_EXTERN_DIR}/rocm-cmake-${rocm_cmake_tag} )
endif( )

include( ROCMSetupVersion )
include( ROCMCreatePackage )
include( ROCMInstallTargets )
include( ROCMPackageConfigHelpers )
include( ROCMInstallSymlinks )

rocm_setup_version( VERSION 0.6.0.0 )

# Append our library helper cmake path and the cmake path for hip (for convenience)
# Users may override HIP path by specifying their own in CMAKE_MODULE_PATH
list( APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake )

# NOTE:  workaround until hcc & hip cmake modules fixes symlink logic in their config files; remove when fixed
list( APPEND CMAKE_PREFIX_PATH /opt/rocm/hcc /opt/rocm/hip )

# Building tensile can add significant compile time; this option allows to build
# library without tensile to allow for rapid iteration without GEMM functionality
option( BUILD_WITH_TENSILE "Building rocBLAS with Tensile or not" ON )
option( BUILD_VERBOSE "Output additional build information" OFF )

# BUILD_SHARED_LIBS is a cmake built-in; we make it an explicit option such that it shows in cmake-gui
option( BUILD_SHARED_LIBS "Build rocBLAS as a shared library" ON )

# Find HCC/HIP dependencies
if( CMAKE_CXX_COMPILER MATCHES ".*/hcc$" )
  find_package( hcc REQUIRED CONFIG PATHS /opt/rocm )
elseif( CMAKE_CXX_COMPILER MATCHES ".*/hipcc$" )
  find_package( hip REQUIRED CONFIG PATHS /opt/rocm )
else( )
  message( AUTHOR_WARNING "rocblas library probably won't build without either the hcc or hipcc compilers. Set the CXX environment variable to specify your desired compiler" )
endif( )

if( BUILD_WITH_TENSILE )
  set( Tensile_LOGIC "FijiROCm15" CACHE STRING "Tensile to use which logic?")
  set_property( CACHE Tensile_LOGIC PROPERTY STRINGS FijiROCm15 )

  option( Tensile_MERGE_FILES "Tensile to merge kernels and solutions files?" ON )
  option( Tensile_SHORT_FILENAMES "Tensile to use short file names? Use if compiler complains they're too long." OFF )
  option( Tensile_PRINT_DEBUG "Tensile to print runtime debug info?" OFF )

  set( tensile_tag "v3.0.4" CACHE STRING "Tensile tag to download" )
  file( DOWNLOAD https://github.com/ROCmSoftwarePlatform/Tensile/archive/${tensile_tag}.tar.gz
    ${PROJECT_EXTERN_DIR}/tensile-${tensile_tag}.tar.gz
    #EXPECTED_HASH SHA256=BC564450B5AE0DFF9C7D15283F0DC1503AF46AE40869AC6E94B9C088FF850F9C # v3.0.0
    EXPECTED_HASH SHA256=b9122f316bb7c6ed2aa17459a5765e6c162325c000202f392e0d415a7fcc862a # v3.0.4
  )

  execute_process( COMMAND ${CMAKE_COMMAND} -E tar xzf ${PROJECT_EXTERN_DIR}/tensile-${tensile_tag}.tar.gz
    WORKING_DIRECTORY ${PROJECT_EXTERN_DIR} )

  # Remove the 'v' prefix from tag to stitch PATH for tensile root
  string( SUBSTRING ${tensile_tag} 1 -1 tensile_tag )
  set( Tensile_ROOT "${PROJECT_EXTERN_DIR}/Tensile-${tensile_tag}" CACHE STRING "Local path of Tensile" )
endif()

# CMake list of machine targets
set( AMDGPU_TARGETS gfx803;gfx900 CACHE STRING "List of specific machine types for library to target" )

add_subdirectory( library )

include( clients/cmake/build-options.cmake )

# Build clients of the library
if( BUILD_CLIENTS_SAMPLES OR BUILD_CLIENTS_TESTS OR BUILD_CLIENTS_BENCHMARKS )
  add_subdirectory( clients )
endif( )
