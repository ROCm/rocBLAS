# ##############################################################################
# Copyright (C) 2025 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# ##############################################################################

set(TEMP_HOST_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../library)
set(_CMAKE_CURRENT_SOURCE_DIR ${TEMP_HOST_SOURCE_DIR})

cmake_minimum_required(VERSION 3.25.2)
project(
  host-library
  VERSION 4.5.0
  LANGUAGES CXX)

if(ROCBLAS_ENABLE_FORTRAN)
  enable_language(Fortran)
endif()

add_library(rocblas-interface INTERFACE)

target_include_directories(
  rocblas-interface
  INTERFACE
    $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/include>
    # This is marked 'internal', but needs to be public for the clients
    $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/include/internal>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    # TODO: Below include may not be required?
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include/rocblas>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include/rocblas/internal>
    $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/src>
    $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/src/include>
    $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/src/include/internal>
    $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/src/src64>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

target_compile_definitions(
  rocblas-interface INTERFACE ROCM_USE_FLOAT16 ROCBLAS_INTERNAL_API
                              ROCBLAS_BETA_FEATURES_API)

# NOTE: We assume separate architectures and lazy library loading are always
# enabled.
if(ROCBLAS_ENABLE_TENSILE)
  target_compile_definitions(
    rocblas-interface
    INTERFACE BUILD_WITH_TENSILE ROCBLAS_TENSILE_SEPARATE_ARCH=1
              ROCBLAS_TENSILE_LAZY_LOAD=1)
endif()

if(ROCBLAS_ENABLE_HIPBLASLT)
  target_link_libraries(rocblas INTERFACE roc::hipblaslt)
  target_compile_definitions(rocblas-interface INTERFACE BUILD_WITH_HIPBLASLT)
endif()

target_compile_features(rocblas-interface INTERFACE cxx_std_17)

target_link_libraries(rocblas-interface INTERFACE hip::host hip::device)

add_library(rocblas-helper OBJECT)

target_link_libraries(rocblas-helper PRIVATE rocblas-interface hip::host
                                             hip::device)

if(ROCBLAS_BUILD_SHARED)
  set_property(TARGET rocblas-helper PROPERTY POSITION_INDEPENDENT_CODE ON)
endif()

add_library(rocblas64 OBJECT)

if(ROCBLAS_BUILD_SHARED)
  set_property(TARGET rocblas64 PROPERTY POSITION_INDEPENDENT_CODE ON)
endif()

if(ROCBLAS_ENABLE_ASAN)
  rocblas_target_configure_sanitizers(rocblas PRIVATE)
endif()

target_compile_definitions(rocblas64 PRIVATE ROCBLAS_INTERNAL_ILP64)

target_link_libraries(rocblas64 PRIVATE rocblas-interface)

if(ROCBLAS_BUILD_SHARED)
  add_library(rocblas SHARED)
else()
  add_library(rocblas STATIC)
endif()

rocm_set_soversion(rocblas ${rocblas_SOVERSION})
add_library(roc::rocblas ALIAS rocblas)

target_include_directories(
  rocblas
  PUBLIC $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/include>
         # This is marked 'internal', but needs to be public for the clients
         $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/include/internal>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
         # TODO: Below include may not be required?
         $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include/rocblas>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include/rocblas/internal>
         $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/rocblas>)

if(ROCBLAS_ENABLE_ASAN)
  rocblas_target_configure_sanitizers(rocblas PRIVATE)
endif()

if(ROCBLAS_BUILD_SHARED)
  target_link_libraries(rocblas PRIVATE roctx64)
endif()

set_target_properties(rocblas PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                         ${PROJECT_BINARY_DIR}/staging)

target_link_libraries(rocblas PRIVATE rocblas-interface rocblas64
                                      OpenMP::OpenMP_CXX)

if(ROCBLAS_ENABLE_FORTRAN)
  add_library(rocblas-fortran OBJECT)
  target_sources(
    rocblas-fortran
    PUBLIC
      $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/include/rocblas_module.f90>
  )
endif()

add_subdirectory(include)
add_subdirectory(src)

# TODO: Fix the source code that uses inconsistent include paths and then remove
# the `configure_file` calls below. This is improper practice and allows clients
# to use wrong (or at least inconsistent) include paths. Refer to file:
# testing_gemm_batched_ex.hpp, which uses the following include
# ~~~
# #include <rocblas/internal/rocblas-beta.h>`
# ~~~
configure_file("${_CMAKE_CURRENT_SOURCE_DIR}/include/internal/rocblas-beta.h"
               "${PROJECT_BINARY_DIR}/include/rocblas/internal" COPYONLY)

configure_file("${_CMAKE_CURRENT_SOURCE_DIR}/include/rocblas.h"
               "${PROJECT_BINARY_DIR}/include/rocblas" COPYONLY)
