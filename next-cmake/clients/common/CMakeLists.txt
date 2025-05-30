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

set(_CMAKE_CURRENT_SOURCE_DIR ${TEMP_CLIENTS_SOURCE_DIR}/common)

add_library(rocblas-clients-common OBJECT)

if(WARN_NOT_ILP64_PREFERRED)
  target_compile_definitions(rocblas-clients-common
                             PRIVATE ROCBLAS_REFERENCE_LIB_WARN)
endif()
target_compile_definitions(
  rocblas-clients-common
  PUBLIC ROCBLAS_REFERENCE_LIB=${BLAS_LIBRARY} ROCM_USE_FLOAT16
         ROCBLAS_INTERNAL_API ROCBLAS_NO_DEPRECATED_WARNINGS
         ROCBLAS_TENSILE_SEPARATE_ARCH=1 ROCBLAS_TENSILE_LAZY_LOAD=1
  PRIVATE GOOGLE_TEST)

target_include_directories(
  rocblas-clients-common
  PUBLIC
    $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/../include/blas1>
    $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/../include/blas2>
    $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/../include/blas3>
    $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/../include/blas_ex>
    # Do not include /blas3, /blas2, /blas1 as they are not needed
    $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/blas_ex>
    $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/../../library/include>
    $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/../../library/src/include>
    $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/../../library/src>)

if(ROCBLAS_ENABLE_ASAN)
  rocblas_target_configure_sanitizers(rocblas-clients-common PRIVATE)
endif()

if(rocm_smi_FOUND)
  target_link_libraries(rocblas-clients-common PUBLIC rocm_smi64)
endif()
target_link_libraries(
  rocblas-clients-common
  PUBLIC roc::rocblas
         hip::host
         hip::device
         BLIS::BLIS
         OpenMP::OpenMP_CXX
         Threads::Threads
         GTest::gtest)

if(ROCBLAS_ENABLE_FORTRAN)
  add_library(rocblas-fortran-clients-common OBJECT)

  if(ROCBLAS_ENABLE_TENSILE)
    target_sources(
      rocblas-fortran-clients-common
      PUBLIC
        ${_CMAKE_CURRENT_SOURCE_DIR}/../include/rocblas_fortran_tensile.f90)
  endif()

  target_sources(
    rocblas-fortran-clients-common
    PUBLIC
      ${_CMAKE_CURRENT_SOURCE_DIR}/../include/rocblas_fortran.f90)

  target_link_libraries(rocblas-fortran-clients-common
    PUBLIC
      rocblas-fortran)

  # Make the object library available to the main client library
  set_target_properties(rocblas-fortran-clients-common
    PROPERTIES
      POSITION_INDEPENDENT_CODE ON)

  target_link_libraries(rocblas-clients-common
    PRIVATE
      rocblas-fortran-clients-common)
else()
  target_compile_definitions(rocblas-clients-common PUBLIC CLIENTS_NO_FORTRAN)
endif()

add_subdirectory(include)
add_subdirectory(src)
