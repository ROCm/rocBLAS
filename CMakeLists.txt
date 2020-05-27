# ########################################################################
# Copyright 2016-2020 Advanced Micro Devices, Inc.
# ########################################################################

# The ROCm platform requires Ubuntu 16.04 or Fedora 24, which has cmake 3.5
cmake_minimum_required( VERSION 3.5 )

# We use C++14 features, this will add compile option: -std=c++14
set( CMAKE_CXX_STANDARD 14 )
# Without this line, it will add -std=gnu++14 instead, which has some issues.
set( CMAKE_CXX_EXTENSIONS OFF )

# This has to be initialized before the project() command appears
# Set the default of CMAKE_BUILD_TYPE to be release, unless user specifies with -D.  MSVC_IDE does not use CMAKE_BUILD_TYPE
if( NOT DEFINED CMAKE_CONFIGURATION_TYPES AND NOT DEFINED CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." )
endif()

# Honor per-config flags in try_compile() source-file signature. cmake v3.7 and up
if( POLICY CMP0066 )
  cmake_policy( SET CMP0066 NEW )
endif( )

if ( NOT DEFINED CMAKE_Fortran_COMPILER AND NOT DEFINED ENV{FC} )
  set( CMAKE_Fortran_COMPILER  "gfortran" )
endif()

unset(ENV{ROCM_BUILD_ID})
project( rocblas LANGUAGES CXX Fortran )

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
# can use the export config files from hip & hcc for both ROCm & nvcc backends.
# ########################################################################

# ########################################################################
# Main
# ########################################################################

if( CMAKE_CXX_COMPILER MATCHES ".*/hipcc$" OR CMAKE_CXX_COMPILER MATCHES ".*/hcc$")
  # Determine if CXX Compiler is hcc, hip-clang or nvcc
  execute_process(COMMAND ${CMAKE_CXX_COMPILER} "--version" OUTPUT_VARIABLE CXX_OUTPUT
                  OUTPUT_STRIP_TRAILING_WHITESPACE
                  ERROR_STRIP_TRAILING_WHITESPACE)
  string(REGEX MATCH "[A-Za-z]* ?clang version" TMP_CXX_VERSION ${CXX_OUTPUT})
  string(REGEX MATCH "[A-Za-z]+" CXX_VERSION_STRING ${TMP_CXX_VERSION})
endif()

if( CXX_VERSION_STRING MATCHES "clang" )
  message( STATUS "Use hip-clang to build for amdgpu backend" )
# set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xclang -fallow-half-arguments-and-returns" )
  set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__HIP_HCC_COMPAT_MODE__=1" )
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
elseif( CXX_VERSION_STRING MATCHES "HCC" )
  message( STATUS "HCC compiler set; ROCm backend selected [ CXX=/opt/rocm-<ver>/bin/hcc cmake ... ]" )
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -hc")

endif( )

# This finds the rocm-cmake project, and installs it if not found
# rocm-cmake contains common cmake code for rocm projects to help setup and install
set( PROJECT_EXTERN_DIR ${CMAKE_CURRENT_BINARY_DIR}/extern )
find_package( ROCM CONFIG QUIET PATHS ${ROCM_PATH} /opt/rocm )
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

  find_package( ROCM REQUIRED CONFIG PATHS ${PROJECT_EXTERN_DIR}/rocm-cmake-${rocm_cmake_tag} )
endif( )

include( ROCMSetupVersion )
include( ROCMCreatePackage )
include( ROCMInstallTargets )
include( ROCMPackageConfigHelpers )
include( ROCMInstallSymlinks )

include (cmake/os-detection.cmake)
get_os_id(OS_ID)
message (STATUS "OS detected is ${OS_ID}")

# Versioning via rocm-cmake
set ( VERSION_STRING "2.23.0" )
rocm_setup_version( VERSION ${VERSION_STRING} )

# Append our library helper cmake path and the cmake path for hip (for convenience)
# Users may override HIP path by specifying their own in CMAKE_MODULE_PATH
list( APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake )

# NOTE:  workaround until hcc & hip cmake modules fixes symlink logic in their config files; remove when fixed
list( APPEND CMAKE_PREFIX_PATH ${ROCM_PATH}/hcc ${ROCM_PATH}/hip /opt/rocm/hcc /opt/rocm/hip )

option( BUILD_VERBOSE "Output additional build information" OFF )

# BUILD_SHARED_LIBS is a cmake built-in; we make it an explicit option such that it shows in cmake-gui
option( BUILD_SHARED_LIBS "Build rocBLAS as a shared library" ON )

option( BUILD_TESTING "Build tests for rocBLAS" ON )

include( clients/cmake/build-options.cmake )

# force library install path to lib (CentOS 7 defaults to lib64)
set(CMAKE_INSTALL_LIBDIR "lib" CACHE INTERNAL "Installation directory for libraries" FORCE)

option( BUILD_WITH_TENSILE_HOST "Use the new tensile client for gemm?" ON )

if( BUILD_WITH_TENSILE )
  set( Tensile_LOGIC "asm_full" CACHE STRING "Tensile to use which logic?")
  set( Tensile_ARCHITECTURE "all" CACHE STRING "Tensile to use which architecture?")
  set( Tensile_CODE_OBJECT_VERSION "V2" CACHE STRING "Tensile code_object_version")
  set( Tensile_COMPILER "hcc" CACHE STRING "Tensile compiler")

  option( Tensile_MERGE_FILES "Tensile to merge kernels and solutions files?" ON )
  option( Tensile_SHORT_FILENAMES "Tensile to use short file names? Use if compiler complains they're too long." OFF )
  option( Tensile_PRINT_DEBUG "Tensile to print runtime debug info?" OFF )

  set( Tensile_TEST_LOCAL_PATH "" CACHE PATH "Use local Tensile directory instead of fetching a GitHub branch" )
  set_property( CACHE Tensile_ARCHITECTURE PROPERTY STRINGS all gfx803 gfx900 gfx906 gfx908)
  set_property( CACHE Tensile_LOGIC PROPERTY STRINGS asm_full asm_lite asm_miopen hip_lite other )
  set_property( CACHE Tensile_CODE_OBJECT_VERSION PROPERTY STRINGS V2 V3 )
  set_property( CACHE Tensile_COMPILER PROPERTY STRINGS hcc hipcc)

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
    find_package(Tensile 4.19.0 EXACT REQUIRED HIP LLVM OpenMP PATHS "${INSTALLED_TENSILE_PATH}")
  endif()
endif()

# Find HCC/HIP dependencies
if( CMAKE_CXX_COMPILER MATCHES ".*/hcc$" )
  find_package( hcc REQUIRED CONFIG PATHS ${ROCM_PATH} /opt/rocm )
  find_package( hip REQUIRED CONFIG PATHS ${ROCM_PATH} /opt/rocm )
endif( )
if( CMAKE_CXX_COMPILER MATCHES ".*/hipcc$" )
  find_package( hip REQUIRED CONFIG PATHS ${ROCM_PATH} /opt/rocm )
endif( )

# CMake list of machine targets
if( Tensile_ARCHITECTURE )
  if( Tensile_ARCHITECTURE STREQUAL "all" )
    set( gpus "gfx803;gfx900;gfx906;gfx908")
  else()
    set( gpus ${Tensile_ARCHITECTURE})
  endif()
  # We must match tensile if architecture is set
  set( ROCBLAS_GPUS ${gpus} )
else()
  set( ROCBLAS_GPUS ${AMDGPU_TARGETS} )
endif( )
message(STATUS "Specific GPU TARGET(s): " ${ROCBLAS_GPUS})

add_subdirectory( library )

# Build clients of the library
if( BUILD_CLIENTS_SAMPLES OR BUILD_CLIENTS_TESTS OR BUILD_CLIENTS_BENCHMARKS )
  add_subdirectory( clients )
endif( )
