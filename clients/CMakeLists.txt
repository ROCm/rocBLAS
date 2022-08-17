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

add_definitions(-D_ROCBLAS_INTERNAL_BFLOAT16_)

# This has to be initialized before the project() command appears
# Set the default of CMAKE_BUILD_TYPE to be release, unless user specifies with -D.  MSVC_IDE does not use CMAKE_BUILD_TYPE
if( NOT DEFINED CMAKE_CONFIGURATION_TYPES AND NOT DEFINED CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." )
endif()

if ( NOT DEFINED CMAKE_Fortran_COMPILER AND NOT DEFINED ENV{FC} )
  set( CMAKE_Fortran_COMPILER "gfortran" )
endif()

# This project may compile dependencies for clients
if ( BUILD_FORTRAN_CLIENTS )
  set( fortran_language "Fortran" )
endif()
project( rocblas-clients LANGUAGES CXX C ${fortran_language} )

if ( BUILD_FORTRAN_CLIENTS )
  set(rocblas_f90_source_clients
      include/rocblas_fortran.f90
  )

  # Set Fortran module output directory
  set(CMAKE_Fortran_MODULE_DIRECTORY ${PROJECT_BINARY_DIR}/include/rocblas/internal)

  # Create rocBLAS Fortran module
  add_library(rocblas_fortran OBJECT ../library/include/rocblas_module.f90)

  if( BUILD_WITH_TENSILE )
    list( APPEND rocblas_f90_source_clients include/rocblas_fortran_tensile.f90 )
  endif()

  add_library(rocblas_fortran_client STATIC ${rocblas_f90_source_clients} $<TARGET_OBJECTS:rocblas_fortran>)
else()
  set( rocblas_fortran_client "")
endif()

if( SKIP_LIBRARY )
  include_directories(${ROCBLAS_LIBRARY_DIR}/include)
  include_directories(${ROCBLAS_LIBRARY_DIR}/include/internal)
else()
  include_directories(${CMAKE_BINARY_DIR}/include/rocblas)
  include_directories(${CMAKE_BINARY_DIR}/include/rocblas/internal)
endif()

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# if it fails to find OpenMP compile and link flags in strange configurations it can just use non-parallel reference computation
# if there is no omp.h to find the client compilation will fail and this should be obvious, used to be REQUIRED
find_package(OpenMP)

if (TARGET OpenMP::OpenMP_CXX)
  set( COMMON_LINK_LIBS "OpenMP::OpenMP_CXX")
  list( APPEND COMMON_LINK_LIBS "-L${HIP_CLANG_ROOT}/lib;-Wl,-rpath=${HIP_CLANG_ROOT}/lib")
endif()

if (TARGET Threads::Threads)
  list( APPEND COMMON_LINK_LIBS "Threads::Threads")
endif()

message(STATUS "CLIENT COMMON CXX_OPTIONS: ${COMMON_CXX_OPTIONS}")
message(STATUS "CLIENT COMMON LINK: ${COMMON_LINK_LIBS}")

list( APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake )

include( client-build-options )

# This option only works for make/nmake and the ninja generators, but no reason it shouldn't be on all the time
# This tells cmake to create a compile_commands.json file that can be used with clang tooling or vim
set( CMAKE_EXPORT_COMPILE_COMMANDS ON )

if( NOT TARGET rocblas )
  find_package( rocblas REQUIRED CONFIG PATHS ${ROCM_PATH} /opt/rocm ${ROCM_PATH}/rocblas /opt/rocm/rocblas ${ROCBLAS_LIBRARY_DIR})
endif( )

# Hip headers required of all clients; clients use hip to allocate device memory
list( APPEND CMAKE_PREFIX_PATH ${ROCM_PATH} /opt/rocm )
if ( NOT hip_FOUND )
  find_package( hip REQUIRED CONFIG PATHS ${ROCM_PATH} )
endif( )

if( BUILD_CLIENTS_SAMPLES )
  add_subdirectory( samples )
endif( )

if( BUILD_CLIENTS_BENCHMARKS OR BUILD_CLIENTS_TESTS)
  if ( NOT WIN32 )
    if (LINK_BLIS)
      set( BLIS_INCLUDE_DIR ${BUILD_DIR}/deps/blis/include/blis )
      set( BLIS_CPP ../common/blis_interface.cpp )
      set( BLAS_LIBRARY ${BUILD_DIR}/deps/blis/lib/libblis.a )
    else()
      set( BLAS_LIBRARY "blas" )
    endif()
  else() # WIN32
    set( BLAS_INCLUDE_DIR ${OPENBLAS_DIR}/include CACHE PATH "OpenBLAS library include path" )
    find_library( BLAS_LIBRARY libopenblas
                    PATHS ${OPENBLAS_DIR}/lib
                    REQUIRED
                    NO_DEFAULT_PATH
                )
  endif()

  # common source files used in subdirectories benchmarks and gtest thus ../common
  set( rocblas_test_bench_common
      ../common/singletons.cpp
      ../common/utility.cpp
      ../common/cblas_interface.cpp
      ../common/rocblas_arguments.cpp
      ../common/argument_model.cpp
      ../common/rocblas_random.cpp
      ../common/rocblas_parse_data.cpp
      ../common/host_alloc.cpp
      ${BLIS_CPP}
    )

  if( BUILD_CLIENTS_BENCHMARKS )
    add_subdirectory( benchmarks )
  endif( )

  if( BUILD_CLIENTS_TESTS )
    add_subdirectory( gtest )
  endif( )

endif()

set( ROCBLAS_COMMON "${PROJECT_BINARY_DIR}/staging/rocblas_common.yaml")
add_custom_command( OUTPUT "${ROCBLAS_COMMON}"
                    COMMAND ${CMAKE_COMMAND} -E copy include/rocblas_common.yaml "${ROCBLAS_COMMON}"
                    DEPENDS include/rocblas_common.yaml
                    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" )

set( ROCBLAS_TEMPLATE "${PROJECT_BINARY_DIR}/staging/rocblas_template.yaml")
add_custom_command( OUTPUT "${ROCBLAS_TEMPLATE}"
                    COMMAND ${CMAKE_COMMAND} -E copy include/rocblas_template.yaml "${ROCBLAS_TEMPLATE}"
                    DEPENDS include/rocblas_template.yaml
                    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" )

set( ROCBLAS_SMOKE "${PROJECT_BINARY_DIR}/staging/rocblas_smoke.yaml")
add_custom_command( OUTPUT "${ROCBLAS_SMOKE}"
                    COMMAND ${CMAKE_COMMAND} -E copy include/rocblas_smoke.yaml "${ROCBLAS_SMOKE}"
                    DEPENDS include/rocblas_smoke.yaml
                    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" )

set( ROCBLAS_GENTEST "${PROJECT_BINARY_DIR}/staging/rocblas_gentest.py")
add_custom_command( OUTPUT "${ROCBLAS_GENTEST}"
                    COMMAND ${CMAKE_COMMAND} -E copy common/rocblas_gentest.py "${ROCBLAS_GENTEST}"
                    DEPENDS common/rocblas_gentest.py
                    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" )


add_custom_target( rocblas-common DEPENDS "${ROCBLAS_COMMON}" "${ROCBLAS_TEMPLATE}" "${ROCBLAS_SMOKE}" "${ROCBLAS_GENTEST}" )

rocm_install(
  FILES ${ROCBLAS_COMMON} ${ROCBLAS_TEMPLATE} ${ROCBLAS_SMOKE} ${ROCBLAS_GENTEST}
  DESTINATION "${CMAKE_INSTALL_BINDIR}"
  COMPONENT clients-common
)
