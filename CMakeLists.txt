# ########################################################################
# Copyright (C) 2016-2022 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# ########################################################################

cmake_minimum_required( VERSION 3.16.8 )


# This has to be initialized before the project() command appears
# Set the default of CMAKE_BUILD_TYPE to be release, unless user specifies with -D.  MSVC_IDE does not use CMAKE_BUILD_TYPE
if( NOT DEFINED CMAKE_CONFIGURATION_TYPES AND NOT DEFINED CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." )
endif()

# ROCM_BUILD_ID is added to the package name by rocm-cmake ROCMSetupVersion
unset(ENV{ROCM_BUILD_ID})

project( rocblas LANGUAGES CXX )

# Append our library helper cmake path and the cmake path for hip (for convenience)
list( APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake )
include(build-options) # library and client
include(toolchain-options)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)


# ########################################################################
# Main
# ########################################################################

# This finds the rocm-cmake project, and installs it if not found
# rocm-cmake contains common cmake code for rocm projects to help setup and install
set( PROJECT_EXTERN_DIR ${CMAKE_CURRENT_BINARY_DIR}/extern )
find_package( ROCM 0.7.3 CONFIG QUIET PATHS ${ROCM_PATH} /opt/rocm )
if( NOT ROCM_FOUND )
  set( rocm_cmake_tag "master" CACHE STRING "rocm-cmake tag to download" )
  file( DOWNLOAD https://github.com/RadeonOpenCompute/rocm-cmake/archive/${rocm_cmake_tag}.zip
      ${PROJECT_EXTERN_DIR}/rocm-cmake-${rocm_cmake_tag}.zip STATUS status LOG log)

  list(GET status 0 status_code)
  list(GET status 1 status_string)

  if(NOT status_code EQUAL 0)
    message(FATAL_ERROR "error: downloading
    'https://github.com/RadeonOpenCompute/rocm-cmake/archive/${rocm_cmake_tag}.zip' failed
    status_code: ${status_code}
    status_string: ${status_string}
    log: ${log}
    ")
  endif()

  message(STATUS "downloading... done")

  execute_process( COMMAND ${CMAKE_COMMAND} -E tar xzvf ${PROJECT_EXTERN_DIR}/rocm-cmake-${rocm_cmake_tag}.zip
    WORKING_DIRECTORY ${PROJECT_EXTERN_DIR} )
  execute_process( COMMAND ${CMAKE_COMMAND} -DCMAKE_INSTALL_PREFIX=${PROJECT_EXTERN_DIR}/rocm-cmake .
    WORKING_DIRECTORY ${PROJECT_EXTERN_DIR}/rocm-cmake-${rocm_cmake_tag} )
  execute_process( COMMAND ${CMAKE_COMMAND} --build rocm-cmake-${rocm_cmake_tag} --target install
    WORKING_DIRECTORY ${PROJECT_EXTERN_DIR})

  find_package( ROCM 0.7.3 REQUIRED CONFIG PATHS ${PROJECT_EXTERN_DIR}/rocm-cmake )
endif( )

# rocm-cmake helpers
include( ROCMSetupVersion )
include( ROCMCreatePackage )
include( ROCMInstallTargets )
include( ROCMPackageConfigHelpers )
include( ROCMInstallSymlinks )
include( ROCMCheckTargetIds )
include( ROCMHeaderWrapper )
include( ROCMClients )


include (os-detection)
get_os_id(OS_ID)
message (STATUS "OS detected is ${OS_ID}")

# Versioning via rocm-cmake
set ( VERSION_STRING "2.45.0" )
rocm_setup_version( VERSION ${VERSION_STRING} )

# Users may override HIP path by specifying their own in CMAKE_MODULE_PATH
# NOTE:  workaround until llvm & hip cmake modules fixes symlink logic in their config files; remove when fixed
list( APPEND CMAKE_PREFIX_PATH ${ROCM_PATH}/llvm ${ROCM_PATH} ${ROCM_PATH}/hip /opt/rocm/llvm /opt/rocm /opt/rocm/hip )

# gpu arch configuration
set( AMDGPU_TARGETS "all" CACHE STRING "Compile for which gpu architectures?")
set_property( CACHE AMDGPU_TARGETS PROPERTY STRINGS all gfx803 gfx900 gfx906:xnack- gfx908:xnack- gfx90a:xnack+ gfx90a:xnack- gfx1010 gfx1012 gfx1030 )

# Detect if target ID syntax if supported for default all AMDGPU_TARGETS list
# Sets the AMDGPU_TARGETS with backward compatiblity
if(COMMAND rocm_check_target_ids)
  rocm_check_target_ids(target_list
      TARGETS "gfx803;gfx900;gfx906:xnack-;gfx908:xnack-;gfx90a:xnack+;gfx90a:xnack-;gfx1010;gfx1012;gfx1030"
  )
else()
  # This section is deprecated. Please use rocm_check_target_ids for future use.
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    execute_process(COMMAND ${CMAKE_CXX_COMPILER} "--help"
      OUTPUT_VARIABLE CXX_OUTPUT
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_STRIP_TRAILING_WHITESPACE)
    string(REGEX MATCH ".mcode\-object\-version" TARGET_ID_SUPPORT ${CXX_OUTPUT})
  endif()
  if(TARGET_ID_SUPPORT)
    if(Tensile_LOGIC STREQUAL "aldebaran")
      # Temporary bypass: if logic aldebaran logic is selected, assume aldebaran compiler support exists
      # To be removed for ROCm 4.4
      set(target_list "gfx803;gfx900;gfx906:xnack-;gfx908:xnack-;gfx90a:xnack+;gfx90a:xnack-;gfx1010;gfx1012;gfx1030")
    else()
      set(target_list "gfx803;gfx900;gfx906:xnack-;gfx908:xnack-;gfx1010;gfx1012;gfx1030")
    endif()
  else()
    set(target_list "gfx803;gfx900;gfx906;gfx908")
  endif()
endif()

set(ROCBLAS_ALL_GPUS "${target_list}" CACHE STRING "List of specific machine types for library to target")

# gpu architectures to build for we may FORCE overwrite expanded "all", AMDGPU_TARGETS always defined by install.sh
if (AMDGPU_TARGETS)
  if( AMDGPU_TARGETS STREQUAL "all" )
    set( gpus "${ROCBLAS_ALL_GPUS}")
  else()
    set( gpus "${AMDGPU_TARGETS}")
  endif()
  # must FORCE set this AMDGPU_TARGETS before any find_package( hip ...), via tensile or in this file
  # to override CACHE var and set --offload-arch flags via hip-config.cmake hip::device dependency
  set( AMDGPU_TARGETS "${gpus}" CACHE STRING "AMD GPU targets to compile for" FORCE )
endif()

if(NOT SKIP_LIBRARY)
  if( BUILD_WITH_TENSILE )
    # we will have expanded "all" for tensile to ensure consistency as we have local rules
    set( Tensile_ARCHITECTURE "${AMDGPU_TARGETS}" CACHE STRING "Tensile to use which architecture?" FORCE)

    if (WIN32)
      set( Tensile_ROOT "${CMAKE_BINARY_DIR}/virtualenv/Lib/site-packages/Tensile" )
    endif()

    set( Tensile_TEST_LOCAL_PATH "" CACHE PATH "Use local Tensile directory instead of fetching a GitHub branch" )

    include(virtualenv)

    if (Tensile_TEST_LOCAL_PATH)
      virtualenv_install(${Tensile_TEST_LOCAL_PATH})
      message (STATUS "using local Tensile from ${Tensile_TEST_LOCAL_PATH}, copied to ${Tensile_ROOT}")
    else()
      # Use the virtual-env setup and download package from specified repot:
      set( tensile_fork "ROCmSoftwarePlatform" CACHE STRING "Tensile fork to use" )
      file (STRINGS "tensile_tag.txt" read_tensile_tag)
      set( tensile_tag ${read_tensile_tag} CACHE STRING "Tensile tag to download" )
      virtualenv_install("git+https://github.com/${tensile_fork}/Tensile.git@${tensile_tag}")
      message (STATUS "using GIT Tensile fork=${tensile_fork} from branch=${tensile_tag}")
    endif()
    message(STATUS "Adding ${VIRTUALENV_HOME_DIR} to CMAKE_PREFIX_PATH")
    list(APPEND CMAKE_PREFIX_PATH ${VIRTUALENV_HOME_DIR})
    if (TENSILE_VERSION)
      find_package(Tensile ${TENSILE_VERSION} EXACT REQUIRED HIP LLVM OpenMP PATHS "${INSTALLED_TENSILE_PATH}")
    else()
      find_package(Tensile 4.33.0 EXACT REQUIRED HIP LLVM OpenMP PATHS "${INSTALLED_TENSILE_PATH}")
    endif()
  endif()
endif()

# Find HIP dependencies
if( CMAKE_CXX_COMPILER_ID MATCHES "Clang" )
  find_package( hip REQUIRED CONFIG PATHS ${HIP_DIR} ${ROCM_PATH} /opt/rocm )
endif( )

message(STATUS "Using AMDGPU_TARGETS: ${AMDGPU_TARGETS}")

# setup rocblas defines used for both the library and clients
if( BUILD_WITH_TENSILE )
    list(APPEND TENSILE_DEFINES BUILD_WITH_TENSILE=1)
else()
    list(APPEND TENSILE_DEFINES BUILD_WITH_TENSILE=0)
endif()

if( BUILD_CLIENTS_SAMPLES OR BUILD_CLIENTS_TESTS OR BUILD_CLIENTS_BENCHMARKS )
  if(NOT CLIENTS_OS)
    rocm_set_os_id(CLIENTS_OS)
    string(TOLOWER "${CLIENTS_OS}" CLIENTS_OS)
    rocm_read_os_release(CLIENTS_OS_VERSION VERSION_ID)
  endif()
  message(STATUS "OS: ${CLIENTS_OS} ${CLIENTS_OS_VERSION}")
  set(OPENMP_PKG "libgomp")
  set(GFORTRAN_PKG "gcc-gfortran")
  if(CLIENTS_OS STREQUAL "sles")
    set(OPENMP_PKG "libgomp1")
    set(GFORTRAN_PKG "gcc-fortran")
  endif()
  if(CLIENTS_OS STREQUAL "centos" AND CLIENTS_OS_VERSION EQUAL 7)
    set(GFORTRAN_PKG "devtoolset-7-gcc-gfortran")
  endif()

  set( BUILD_CLIENTS ON )
  rocm_package_setup_component(clients)
  rocm_package_setup_client_component( clients-common )
  if(BUILD_CLIENTS_TESTS)
    rocm_package_setup_client_component(
      tests
      DEPENDS
        COMPONENT clients-common
        DEB "libomp-dev >= 5.0.1"
        RPM "${OPENMP_PKG}"
    )
  endif()
  if(BUILD_CLIENTS_BENCHMARKS)
    rocm_package_setup_client_component(
      benchmarks
      DEPENDS
        COMPONENT clients-common
        DEB "libomp-dev >= 5.0.1"
        RPM "${OPENMP_PKG}"
    )
  endif()
  if(BUILD_FORTRAN_CLIENTS)
    rocm_package_add_rpm_dependencies(COMPONENT tests DEPENDS "${GFORTRAN_PKG}")
    rocm_package_add_deb_dependencies(COMPONENT tests DEPENDS "gfortran >= 7.4.0")
    rocm_package_add_rpm_dependencies(COMPONENT benchmarks DEPENDS "${GFORTRAN_PKG}")
    rocm_package_add_deb_dependencies(COMPONENT benchmarks DEPENDS "gfortran >= 7.4.0")
  endif()
endif()

if( NOT SKIP_LIBRARY )
    add_subdirectory( library )
endif()

# Build clients of the library
if( BUILD_CLIENTS )
  add_subdirectory( clients )
endif( )

# The following code is setting variables to control the behavior of CPack to generate our
if( WIN32 )
    set( CPACK_SOURCE_GENERATOR "ZIP" )
    set( CPACK_GENERATOR "ZIP" )
endif( )

# Package specific CPACK vars
# As of ROCm 3.8, HIP_RUNTIME has been changed from "ROCclr" to "rocclr"
# As of ROCm 3.8, HIP_RUNTIME has been changed from "cuda" to "cudart"
if( HIP_RUNTIME MATCHES ".*cuda.*" )
  rocm_package_add_dependencies(DEPENDS "hip-nvcc >= 3.5.0")
else()
  rocm_package_add_dependencies(DEPENDS "hip-rocclr >= 4.0.0")
endif()

if( OS_ID_sles )
  rocm_package_add_rpm_dependencies(DEPENDS "libLLVM7 >= 7.0.1")
endif()

set( CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.md" )
set( CPACK_RPM_PACKAGE_LICENSE "MIT and BSD")

if (WIN32)
  SET( CMAKE_INSTALL_PREFIX "C:/hipSDK" CACHE PATH "Install path" FORCE )
  SET( INSTALL_PREFIX "C:/hipSDK" )
  SET( CPACK_SET_DESTDIR FALSE )
  SET( CPACK_PACKAGE_INSTALL_DIRECTORY "C:/hipSDK" )
  SET( CPACK_PACKAGING_INSTALL_PREFIX "" )
  set( CPACK_INCLUDE_TOPLEVEL_DIRECTORY OFF )
else()
  if( NOT CPACK_PACKAGING_INSTALL_PREFIX )
    set( CPACK_PACKAGING_INSTALL_PREFIX ${ROCM_PATH} )
  endif()
endif( )

set( CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "\${CPACK_PACKAGING_INSTALL_PREFIX}" "\${CPACK_PACKAGING_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}" "\${CPACK_PACKAGING_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}" )

# Give rocblas compiled for CUDA backend a different name
if( CMAKE_CXX_COMPILER_ID MATCHES "Clang" )
    set( package_name rocblas )
else( )
    set( package_name rocblas-alt )
endif( )

set( ROCBLAS_CONFIG_DIR "\${CPACK_PACKAGING_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}" CACHE PATH "Path placed into ldconfig file" )

rocm_create_package(
    NAME ${package_name}
    DESCRIPTION "rocBLAS is AMD's library for BLAS on ROCm. It is implemented in HIP and optimized for AMD GPUs."
    MAINTAINER "rocBLAS Maintainer <rocblas-maintainer@amd.com>"
    LDCONFIG
    LDCONFIG_DIR ${ROCBLAS_CONFIG_DIR}
)

#
# ADDITIONAL TARGETS FOR CODE COVERAGE
if(BUILD_CODE_COVERAGE)
  #
  # > make coverage_cleanup (clean coverage related files.)
  # > make coverage GTEST_FILTER=<>
  # will run:
  #  > make coverage_analysis GTEST_FILTER=<> (analyze tests)
  #  > make coverage_output (generate html documentation)
  #
  #

  #
  # Run coverage analysis
  #
  add_custom_target(coverage_analysis
    COMMAND echo Coverage GTEST_FILTER=\${GTEST_FILTER}
    COMMAND ./clients/staging/rocblas-test --gtest_filter=\"\${GTEST_FILTER}\"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )

  add_dependencies(coverage_analysis rocblas)

  #
  # Prepare coverage output
  # This little script is generated because the option '--gcov-tool <program name>' of lcov cannot take arguments.
  #
  add_custom_target(coverage_output
    DEPENDS coverage_analysis
    COMMAND mkdir -p lcoverage
    COMMAND echo "\\#!/bin/bash" > llvm-gcov.sh
    COMMAND echo "\\# THIS FILE HAS BEEN GENERATED" >> llvm-gcov.sh
    COMMAND printf "exec /opt/rocm/llvm/bin/llvm-cov gcov $$\\@" >> llvm-gcov.sh
    COMMAND chmod +x llvm-gcov.sh
    )

  #
  # Generate coverage output.
  #
  add_custom_command(TARGET coverage_output
    COMMAND lcov --directory . --base-directory . --gcov-tool ${CMAKE_BINARY_DIR}/llvm-gcov.sh --capture -o lcoverage/raw_main_coverage.info
    COMMAND lcov --remove lcoverage/raw_main_coverage.info "'/opt/*'" "'/usr/*'" -o lcoverage/main_coverage.info
    COMMAND genhtml --ignore-errors source lcoverage/main_coverage.info --output-directory lcoverage
    )

  add_custom_target(coverage DEPENDS coverage_output)

  #
  # Coverage cleanup
  #
  add_custom_target(coverage_cleanup
    COMMAND find ${CMAKE_BINARY_DIR} -name *.gcda -delete
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
endif()
