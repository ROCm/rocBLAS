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

# Append our library helper cmake path and the cmake path for hip (for convenience)
# Users may override HIP path by specifying their own in CMAKE_MODULE_PATH
list( APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake )

# include( build-version )
# project_version( NAME rocblas LANGUAGES CXX )
project( rocblas VERSION 0.5.4.0 LANGUAGES CXX )

# Building tensile can add significant compile time; this option allows to build
# library without tensile to allow for rapid iteration without GEMM functionality
option( BUILD_WITH_TENSILE "Building rocBLAS with Tensile or not" OFF )

option( BUILD_CLIENTS "Build rocBLAS clients" OFF )

# BUILD_SHARED_LIBS is a cmake built-in; we make it an explicit option such that it shows in cmake-gui
option( BUILD_SHARED_LIBS "Build rocBLAS as a shared library" ON )

# Find HIP dependency
find_package( hip REQUIRED CONFIG PATHS /opt/rocm/hip )

if( BUILD_WITH_TENSILE )
  set( Tensile_LOGIC "FijiROCm15" CACHE STRING "Tensile to use which logic?")
  set_property( CACHE Tensile_LOGIC PROPERTY STRINGS
    FijiROCm15
    )

  option( Tensile_MERGE_FILES "Tensile to merge kernels and solutions files?" ON )
  option( Tensile_SHORT_FILENAMES "Tensile to use short file names? Use if compiler complains they're too long." OFF )
  option( Tensile_PRINT_DEBUG "Tensile to print runtime debug info?" OFF )

  # defines
  # Tensile_INCLUDE_DIRS
  # TensileLib_LIBRARIES
  # TensileLogger_LIBRARIES
  set( tensile_tag "v3.0.0" CACHE STRING "Tensile tag to download" )
  file( DOWNLOAD https://github.com/ROCmSoftwarePlatform/Tensile/archive/${tensile_tag}.tar.gz
    ${CMAKE_CURRENT_BINARY_DIR}/tensile-${tensile_tag}.tar.gz
    EXPECTED_HASH SHA256=BC564450B5AE0DFF9C7D15283F0DC1503AF46AE40869AC6E94B9C088FF850F9C )

  execute_process( COMMAND ${CMAKE_COMMAND} -E tar xzf ${CMAKE_CURRENT_BINARY_DIR}/tensile-${tensile_tag}.tar.gz
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} )

  set( Tensile_ROOT "${CMAKE_CURRENT_BINARY_DIR}/Tensile-${tensile_tag}" CACHE STRING "Local path of Tensile.")

  if (Tensile_ROOT)
    message(STATUS "Tensile_ROOT=${Tensile_ROOT} specified")
  else()
    find_package(Tensile QUIET)
    if (Tensile_FOUND)
      message(STATUS "Tensile package found.")
    else()
      include(cmake/external-Tensile.cmake )
      list( APPEND rocblas_dependencies Tensile )
      list( APPEND CMAKE_PREFIX_PATH ${Tensile_ROOT} )
      message( STATUS "Tensile not installed; will download to Tensile_ROOT=${Tensile_ROOT}" )
    endif()
  endif()

endif()

add_subdirectory( library )

# Build clients of the library
if( BUILD_CLIENTS )
  include( clients/cmake/build-options.cmake )

  add_subdirectory( clients )
endif( )

# CMake list of machine targets
set( AMDGPU_TARGETS "gfx803;gfx900" CACHE STRING "List of specific machine types for library to target" )

foreach( target ${AMDGPU_TARGET} )
  target_compile_options( rocblas PRIVATE --amdgpu-target=${target} )
endforeach( )