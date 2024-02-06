# ########################################################################
# Copyright (C) 2016-2024 Advanced Micro Devices, Inc. All rights reserved.
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

set(rocblas_bench_source
  client.cpp
  )

add_executable( rocblas-bench ${rocblas_bench_source} ${rocblas_test_bench_common} )

if( BUILD_WITH_TENSILE )
  set(rocblas_gemm_tune_source
    gemm_tune/gemm_tune_client.cpp
    gemm_tune/gemm_tuners.cpp
    )

  add_executable( rocblas-gemm-tune ${rocblas_gemm_tune_source} ${rocblas_test_bench_common} )
endif()

# Internal header includes
target_include_directories( rocblas-bench
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include/blas1>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include/blas2>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include/blas3>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include/blas_ex>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../library/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../library/src/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../library/src>
)
if( BUILD_WITH_TENSILE )
  target_include_directories( rocblas-gemm-tune
    PRIVATE
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include/blas3>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../library/include>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../library/src/include>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../library/src>
  )
endif()

# External header includes included as system files
target_include_directories( rocblas-bench
  SYSTEM PRIVATE
    $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${BLAS_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${BLIS_INCLUDE_DIR}> # may be blank if not used
)
if( BUILD_WITH_TENSILE )
  target_include_directories( rocblas-gemm-tune
    SYSTEM PRIVATE
      $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
      $<BUILD_INTERFACE:${BLAS_INCLUDE_DIR}>
      $<BUILD_INTERFACE:${BLIS_INCLUDE_DIR}> # may be blank if not used
  )
endif()

if( BUILD_FORTRAN_CLIENTS )
  target_link_libraries( rocblas-bench PRIVATE rocblas_fortran_client )
endif()

target_link_libraries( rocblas-bench PRIVATE ${BLAS_LIBRARY} roc::rocblas )
if( BUILD_WITH_TENSILE )
  target_link_libraries( rocblas-gemm-tune PRIVATE ${BLAS_LIBRARY} roc::rocblas )
endif()

if( CUDA_FOUND )
  target_include_directories( rocblas-bench
    PRIVATE
      $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>
      $<BUILD_INTERFACE:${hip_INCLUDE_DIRS}>
    )
  if( BUILD_WITH_TENSILE )
    target_include_directories( rocblas-gemm-tune
      PRIVATE
        $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>
        $<BUILD_INTERFACE:${hip_INCLUDE_DIRS}>
      )
  endif()
  target_compile_definitions( rocblas-bench PRIVATE __HIP_PLATFORM_NVCC__ )
  if( BUILD_WITH_TENSILE )
    target_compile_definitions( rocblas-gemm-tune PRIVATE __HIP_PLATFORM_NVCC__ )
  endif()
  target_link_libraries( rocblas-bench PRIVATE ${CUDA_LIBRARIES} )
  if( BUILD_WITH_TENSILE )
    target_link_libraries( rocblas-gemm-tune PRIVATE ${CUDA_LIBRARIES} )
  endif()
else( )
  # auto set in hip_common.h
  #target_compile_definitions( rocblas-bench PRIVATE __HIP_PLATFORM_HCC__ )
  target_link_libraries( rocblas-bench PRIVATE hip::host hip::device )
  if( BUILD_WITH_TENSILE )
    target_link_libraries( rocblas-gemm-tune PRIVATE hip::host hip::device )
  endif()
endif()

if( CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  # GCC or hip-clang needs specific flags to turn on f16c intrinsics
  target_compile_options( rocblas-bench PRIVATE -mf16c )
  if( BUILD_WITH_TENSILE )
    target_compile_options( rocblas-gemm-tune PRIVATE -mf16c )
  endif()
endif()

target_compile_definitions( rocblas-bench PRIVATE ROCBLAS_BENCH ROCM_USE_FLOAT16 ROCBLAS_INTERNAL_API ROCBLAS_NO_DEPRECATED_WARNINGS ${TENSILE_DEFINES} )
if( BUILD_WITH_TENSILE )
  target_compile_definitions( rocblas-gemm-tune PRIVATE ROCBLAS_BENCH ROCM_USE_FLOAT16 ROCBLAS_INTERNAL_API ROCBLAS_NO_DEPRECATED_WARNINGS ${TENSILE_DEFINES} )
endif()
if ( NOT BUILD_FORTRAN_CLIENTS )
  target_compile_definitions( rocblas-bench PRIVATE CLIENTS_NO_FORTRAN )
  if( BUILD_WITH_TENSILE )
    target_compile_definitions( rocblas-gemm-tune PRIVATE CLIENTS_NO_FORTRAN )
  endif()
endif()

target_compile_options(rocblas-bench PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${COMMON_CXX_OPTIONS}>)
if( BUILD_WITH_TENSILE )
  target_compile_options(rocblas-gemm-tune PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${COMMON_CXX_OPTIONS}>)
endif()
# target_compile_options does not go to linker like CMAKE_CXX_FLAGS does, so manually add
if (NOT WIN32)
  list( APPEND COMMON_LINK_LIBS "-lm -lstdc++fs")
  if (NOT BUILD_FORTRAN_CLIENTS)
    list( APPEND COMMON_LINK_LIBS "-lgfortran") # for lapack
  endif()
else()
  list( APPEND COMMON_LINK_LIBS "libomp")
endif()
target_link_libraries( rocblas-bench PRIVATE ${COMMON_LINK_LIBS} )
if( BUILD_WITH_TENSILE )
  target_link_libraries( rocblas-gemm-tune PRIVATE ${COMMON_LINK_LIBS} )
endif()

set_target_properties( rocblas-bench PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
if( BUILD_WITH_TENSILE )
  set_target_properties( rocblas-gemm-tune PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
endif()

add_dependencies( rocblas-bench rocblas-common )
if( BUILD_WITH_TENSILE )
  add_dependencies( rocblas-gemm-tune rocblas-common )
endif()

add_subdirectory ( ./perf_script )

rocm_install(TARGETS rocblas-bench COMPONENT benchmarks)
if( BUILD_WITH_TENSILE )
  rocm_install(TARGETS rocblas-gemm-tune COMPONENT benchmarks)
endif()
