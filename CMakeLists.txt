# ########################################################################
# Copyright (C) 2016-2023 Advanced Micro Devices, Inc. All rights reserved.
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

# This will add compile option: -std=c++17
set( CMAKE_CXX_STANDARD 17 )
# Without this line, it will add -std=gnu++17 instead, which may have issues.
set( CMAKE_CXX_EXTENSIONS OFF )

if (NOT python)
  set(python "python3") # default for linux
endif()

# This has to be initialized before the project() command appears
# Set the default of CMAKE_BUILD_TYPE to be release, unless user specifies with -D.  MSVC_IDE does not use CMAKE_BUILD_TYPE
if( NOT DEFINED CMAKE_CONFIGURATION_TYPES AND NOT DEFINED CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." )
endif()

# disable this next option and test build whenever compiler changes to verify unused arguments are safe to ignore
add_compile_options( -Wno-unused-command-line-argument )

# TODO: address [[nodiscard]] warnings
add_compile_options( $<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-result> )


# CMake v3.21.0 can identify ROCMClang, but there are some issues with feature detection.
# For now, disable ROCMClang detection to make it work the same as CMake v3.20 and earlier.
set(__skip_rocmclang ON)

unset(ENV{ROCM_BUILD_ID})
project( rocblas LANGUAGES CXX )

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# ########################################################################
# NOTE:  CUDA compiling path
# ########################################################################
# I have tried compiling rocBLAS library source with multiple methods,
# and ended up using the approach where we set the CXX compiler to hipcc.
# I didn't like using the HIP_ADD_LIBRARY or CUDA_ADD_LIBRARY approaches,
# for the reasons I list here.
# 1.  Adding header include directories is through HIP_INCLUDE_DIRECTORIES(), which
# is global to a directory and affects all targets
# 2.  You must add HIP_SOURCE_PROPERTY_FORMAT OBJ properties to .cpp files
# to get HIP_ADD_LIBRARY to recognize the file
# 3.  HIP_ADD_LIBRARY invokes a call to add_custom_command() to compile files,
# and rocBLAS does the same.  The order in which custom commands execute is
# undefined, and sometimes a file is attempted to be compiled before it has
# been generated.  The fix for this is to create 'PHONY' targets, which I
# don't desire.

# Using hipcc allows us to avoid the above problems, with two primary costs:
# 1.  The cmake logic to detect compiler features fails with nvcc backend
# 2.  Upfront cost to figure out all the strange compiler/linker flags I define
# below.

# Hopefully, cost #2 is already paid.  All in all, I want to get rid of the
# need for hipcc, and hope that at some point of time in the future we
# can use the export config files from hip for both ROCm & nvcc backends.
# ########################################################################

# ########################################################################
# Main
# ########################################################################

if( CMAKE_CXX_COMPILER_ID MATCHES "Clang" )
  # Determine if CXX Compiler is hip-clang or nvcc
  execute_process(COMMAND ${CMAKE_CXX_COMPILER} "--version" OUTPUT_VARIABLE CXX_OUTPUT
                  OUTPUT_STRIP_TRAILING_WHITESPACE
                  ERROR_STRIP_TRAILING_WHITESPACE)
  string(REGEX MATCH "[A-Za-z]* ?clang version" TMP_CXX_VERSION ${CXX_OUTPUT})
  string(REGEX MATCH "[A-Za-z]+" CXX_VERSION_STRING ${TMP_CXX_VERSION})
endif()

if( CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  message( STATUS "Use hip-clang to build for amdgpu backend" )
# set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xclang -fallow-half-arguments-and-returns" )
  set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__HIP_HCC_COMPAT_MODE__=1" )
  if ( CMAKE_BUILD_TYPE MATCHES "Debug" AND NOT WIN32)
    set ( CMAKE_CXX_FLAGS_DEBUG "-O1 ${CMAKE_CXX_FLAGS_DEBUG} -gsplit-dwarf -ggdb" )
  endif()
elseif( CXX_VERSION_STRING MATCHES "nvcc" )
  message( STATUS "HIPCC nvcc compiler detected; CUDA backend selected" )

  set( CMAKE_C_COMPILE_OPTIONS_PIC "-Xcompiler ${CMAKE_C_COMPILE_OPTIONS_PIC}" )
  set( CMAKE_CXX_COMPILE_OPTIONS_PIC "-Xcompiler ${CMAKE_CXX_COMPILE_OPTIONS_PIC}" )
  set( CMAKE_SHARED_LIBRARY_C_FLAGS "-Xlinker ${CMAKE_SHARED_LIBRARY_C_FLAGS}" )
  set( CMAKE_SHARED_LIBRARY_CXX_FLAGS "-Xlinker ${CMAKE_SHARED_LIBRARY_CXX_FLAGS}" )
  set( CMAKE_SHARED_LIBRARY_SONAME_C_FLAG "-Xlinker -soname," )
  set( CMAKE_SHARED_LIBRARY_SONAME_CXX_FLAG "-Xlinker -soname," )
  set( CMAKE_SHARED_LIBRARY_RUNTIME_C_FLAG "-Xlinker -rpath," )
  set( CMAKE_SHARED_LIBRARY_RUNTIME_CXX_FLAG "-Xlinker -rpath," )
  set( CMAKE_EXECUTABLE_RUNTIME_C_FLAG "-Xlinker -rpath," )
  set( CMAKE_EXECUTABLE_RUNTIME_CXX_FLAG "-Xlinker -rpath," )
  set( CMAKE_C_COMPILE_OPTIONS_VISIBILITY "-Xcompiler ${CMAKE_C_COMPILE_OPTIONS_VISIBILITY}" )
  set( CMAKE_CXX_COMPILE_OPTIONS_VISIBILITY "-Xcompiler ${CMAKE_CXX_COMPILE_OPTIONS_VISIBILITY}" )
  set( CMAKE_C_COMPILE_OPTIONS_VISIBILITY_INLINES_HIDDEN "-Xcompiler ${CMAKE_C_COMPILE_OPTIONS_VISIBILITY_INLINES_HIDDEN}" )
  set( CMAKE_CXX_COMPILE_OPTIONS_VISIBILITY_INLINES_HIDDEN "-Xcompiler ${CMAKE_CXX_COMPILE_OPTIONS_VISIBILITY_INLINES_HIDDEN}" )
elseif( CMAKE_CXX_COMPILER MATCHES ".*/hcc$" )
  message( STATUS "ERROR: HCC compiler is no longer supported!" )
endif( )

# This finds the rocm-cmake project, and installs it if not found
# rocm-cmake contains common cmake code for rocm projects to help setup and install
set( PROJECT_EXTERN_DIR ${CMAKE_CURRENT_BINARY_DIR}/extern )
find_package( ROCM 0.6 CONFIG QUIET PATHS ${ROCM_PATH} /opt/rocm )
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

  find_package( ROCM 0.6 REQUIRED CONFIG PATHS ${PROJECT_EXTERN_DIR}/rocm-cmake )
endif( )

include( ROCMSetupVersion )
include( ROCMCreatePackage )
include( ROCMInstallTargets )
include( ROCMPackageConfigHelpers )
include( ROCMInstallSymlinks )
include( ROCMCheckTargetIds OPTIONAL )


include (cmake/os-detection.cmake)
get_os_id(OS_ID)
message (STATUS "OS detected is ${OS_ID}")

# Versioning via rocm-cmake
set ( VERSION_STRING "2.42.0" )
rocm_setup_version( VERSION ${VERSION_STRING} )

# Append our library helper cmake path and the cmake path for hip (for convenience)
# Users may override HIP path by specifying their own in CMAKE_MODULE_PATH
list( APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake )

# NOTE:  workaround until llvm & hip cmake modules fixes symlink logic in their config files; remove when fixed
list( APPEND CMAKE_PREFIX_PATH ${ROCM_PATH}/llvm ${ROCM_PATH}/hip /opt/rocm/llvm /opt/rocm/hip )

option( BUILD_VERBOSE "Output additional build information" OFF )

# BUILD_SHARED_LIBS is a cmake built-in; we make it an explicit option such that it shows in cmake-gui
option( BUILD_SHARED_LIBS "Build rocBLAS as a shared library" ON )

option( BUILD_TESTING "Build tests for rocBLAS" ON )

# FOR OPTIONAL CODE COVERAGE
option(BUILD_CODE_COVERAGE "Build rocBLAS with code coverage enabled" OFF)
if(BUILD_CODE_COVERAGE)
  add_compile_options(-fprofile-arcs -ftest-coverage)
  add_link_options(--coverage)
endif()

include( clients/cmake/build-options.cmake )

# FOR OPTIONAL ADDRESS SANITIZER
option(BUILD_ADDRESS_SANITIZER "Build with address sanitizer enabled" OFF)
if(BUILD_ADDRESS_SANITIZER AND BUILD_SHARED_LIBS)
  # Fortran not supported, add_link_options below invalid for fortran linking
  set(BUILD_FORTRAN_CLIENTS OFF)

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -shared-libasan")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address -shared-libasan")
  add_link_options(-fuse-ld=lld)
endif()

# force library install path to lib (CentOS 7 defaults to lib64)
set(CMAKE_INSTALL_LIBDIR "lib" CACHE INTERNAL "Installation directory for libraries" FORCE)

# gpu arch configuration
set( AMDGPU_TARGETS "all" CACHE STRING "Compile for which gpu architectures?")
set_property( CACHE AMDGPU_TARGETS PROPERTY STRINGS all gfx803 gfx900 gfx906:xnack- gfx908:xnack- gfx90a:xnack+ gfx90a:xnack- gfx1010 gfx1011 gfx1012 gfx1030 )

# Detect if target ID syntax if supported for default all AMDGPU_TARGETS list
# Sets the AMDGPU_TARGETS with backward compatiblity
if(COMMAND rocm_check_target_ids)
  rocm_check_target_ids(target_list
      TARGETS "gfx803;gfx900;gfx906:xnack-;gfx908:xnack-;gfx90a:xnack+;gfx90a:xnack-;gfx1010;gfx1011;gfx1012;gfx1030"
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
      set(target_list "gfx803;gfx900;gfx906:xnack-;gfx908:xnack-;gfx90a:xnack+;gfx90a:xnack-;gfx1010;gfx1011;gfx1012;gfx1030")
    else()
      set(target_list "gfx803;gfx900;gfx906:xnack-;gfx908:xnack-;gfx1010;gfx1011;gfx1012;gfx1030")
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
  option( BUILD_WITH_TENSILE "Build full functionality which requires tensile?" ON )

  if( BUILD_WITH_TENSILE )
    # we will have expanded "all" for tensile to ensure consistency as we have local rules
    set( Tensile_ARCHITECTURE "${AMDGPU_TARGETS}" CACHE STRING "Tensile to use which architecture?" FORCE)

    set( Tensile_LOGIC "asm_full" CACHE STRING "Tensile to use which logic?")
    set( Tensile_CODE_OBJECT_VERSION "V2" CACHE STRING "Tensile code_object_version")
    set( Tensile_COMPILER "hipcc" CACHE STRING "Tensile compiler")
    set( Tensile_LIBRARY_FORMAT "msgpack" CACHE STRING "Tensile library format")

    option( Tensile_MERGE_FILES "Tensile to merge kernels and solutions files?" ON )
    option( Tensile_SHORT_FILENAMES "Tensile to use short file names? Use if compiler complains they're too long." OFF )
    option( Tensile_PRINT_DEBUG "Tensile to print runtime debug info?" OFF )

    set( Tensile_TEST_LOCAL_PATH "" CACHE PATH "Use local Tensile directory instead of fetching a GitHub branch" )

    set_property( CACHE Tensile_LOGIC PROPERTY STRINGS aldebaran asm_full asm_lite asm_miopen hip_lite other )
    set_property( CACHE Tensile_CODE_OBJECT_VERSION PROPERTY STRINGS V2 V3 )
    set_property( CACHE Tensile_COMPILER PROPERTY STRINGS hcc hipcc)
    set_property( CACHE Tensile_LIBRARY_FORMAT PROPERTY STRINGS msgpack yaml)

    if(Tensile_LIBRARY_FORMAT MATCHES "yaml")
      option(TENSILE_USE_LLVM      "Use LLVM for parsing config files." ON)
      option(TENSILE_USE_MSGPACK   "Use msgpack for parsing config files." OFF)
    else()
      option(TENSILE_USE_LLVM      "Use LLVM for parsing config files." OFF)
      option(TENSILE_USE_MSGPACK   "Use msgpack for parsing config files." ON)
    endif()

    if (WIN32)
      set( Tensile_ROOT "${CMAKE_BINARY_DIR}/virtualenv/Lib/site-packages/Tensile" )
    endif()

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
      find_package(Tensile 4.31.0 EXACT REQUIRED HIP LLVM OpenMP PATHS "${INSTALLED_TENSILE_PATH}")
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
  set( BUILD_CLIENTS ON )
endif()

if( NOT SKIP_LIBRARY )
    add_subdirectory( library )
endif()

# Build clients of the library
if( BUILD_CLIENTS )
  add_subdirectory( clients )
endif( )

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

# CMake to build documentation
## Sphinx
set(SPHINX_DIR "docs/sphinx" CACHE PATH "Path to sphinx root directory")
set(SPHINX_BUILD_DIR "_build" CACHE PATH "Path to sphinx build directory")
set(SPHINX_BUILD_TYPE "html" CACHE STRING "Sphinx build type")
set(SPHINX_EXTRA_ARGS "" CACHE STRING "Extra arguments for Sphinx build")
set(
    SPHINX_ARGS
    -T
    -b ${SPHINX_BUILD_TYPE}
    -d ${SPHINX_BUILD_DIR}/doctrees
    -D language=en
    ${SPHINX_EXTRA_ARGS}
)
add_custom_target(
    sphinx_docs
    COMMAND pip3 install -r requirements.txt
    COMMAND python3 -m sphinx ${SPHINX_ARGS} . ${SPHINX_BUILD_DIR}/${SPHINX_BUILD_TYPE}
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/${SPHINX_DIR}
)

## Doxygen
set(DOXYGEN_DIR "docs/doxygen" CACHE PATH "Path to doxygen root directory")
### Any other doxygen arguments can go here
find_package(Doxygen)
if (DOXYGEN_FOUND)
    add_custom_target(
        doxygen_docs
        Doxygen::doxygen Doxyfile
        WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/${DOXYGEN_DIR}
    )
endif()

## All docs phony target
add_custom_target(docs DEPENDS sphinx_docs)
if (DOXYGEN_FOUND)
    add_dependencies(docs doxygen_docs)
endif()
