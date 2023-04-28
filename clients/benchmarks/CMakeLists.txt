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

set(rocblas_bench_source
  client.cpp
  )

add_executable( rocblas-bench ${rocblas_bench_source} ${rocblas_test_bench_common} )
add_executable( rocblas_v3-bench ${rocblas_bench_source} ${rocblas_test_bench_common} )

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
target_include_directories( rocblas_v3-bench
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

# External header includes included as system files
target_include_directories( rocblas-bench
  SYSTEM PRIVATE
    $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${BLAS_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${BLIS_INCLUDE_DIR}> # may be blank if not used
)
target_include_directories( rocblas_v3-bench
  SYSTEM PRIVATE
    $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${BLAS_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${BLIS_INCLUDE_DIR}> # may be blank if not used
)

if( BUILD_FORTRAN_CLIENTS )
  target_link_libraries( rocblas-bench PRIVATE rocblas_fortran_client )
  target_link_libraries( rocblas_v3-bench PRIVATE rocblas_fortran_client )
endif( )

target_link_libraries( rocblas-bench PRIVATE ${BLAS_LIBRARY} roc::rocblas )
target_link_libraries( rocblas_v3-bench PRIVATE ${BLAS_LIBRARY} roc::rocblas )

target_compile_definitions( rocblas_v3-bench PRIVATE ROCBLAS_V3)

if( CUDA_FOUND )
  target_include_directories( rocblas-bench
    PRIVATE
      $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>
      $<BUILD_INTERFACE:${hip_INCLUDE_DIRS}>
    )
  target_include_directories( rocblas_v3-bench
    PRIVATE
      $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>
      $<BUILD_INTERFACE:${hip_INCLUDE_DIRS}>
    )
  target_compile_definitions( rocblas-bench PRIVATE __HIP_PLATFORM_NVCC__ )
  target_compile_definitions( rocblas_v3-bench PRIVATE __HIP_PLATFORM_NVCC__ )
  target_link_libraries( rocblas-bench PRIVATE ${CUDA_LIBRARIES} )
  target_link_libraries( rocblas_v3-bench PRIVATE ${CUDA_LIBRARIES} )
else( )
  # auto set in hip_common.h
  #target_compile_definitions( rocblas-bench PRIVATE __HIP_PLATFORM_HCC__ )
  target_link_libraries( rocblas-bench PRIVATE hip::host hip::device )
  target_link_libraries( rocblas_v3-bench PRIVATE hip::host hip::device )
endif( )

if( CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  # GCC or hip-clang needs specific flags to turn on f16c intrinsics
  target_compile_options( rocblas-bench PRIVATE -mf16c )
  target_compile_options( rocblas_v3-bench PRIVATE -mf16c )
endif( )

target_compile_definitions( rocblas-bench PRIVATE ROCBLAS_BENCH ROCM_USE_FLOAT16 ROCBLAS_INTERNAL_API ROCBLAS_NO_DEPRECATED_WARNINGS ${TENSILE_DEFINES} )
target_compile_definitions( rocblas_v3-bench PRIVATE ROCBLAS_BENCH ROCM_USE_FLOAT16 ROCBLAS_INTERNAL_API ROCBLAS_NO_DEPRECATED_WARNINGS ${TENSILE_DEFINES} )
if ( NOT BUILD_FORTRAN_CLIENTS )
  target_compile_definitions( rocblas-bench PRIVATE CLIENTS_NO_FORTRAN )
  target_compile_definitions( rocblas_v3-bench PRIVATE CLIENTS_NO_FORTRAN )
endif()

target_compile_options(rocblas-bench PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${COMMON_CXX_OPTIONS}>)
target_compile_options(rocblas_v3-bench PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${COMMON_CXX_OPTIONS}>)
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
target_link_libraries( rocblas_v3-bench PRIVATE ${COMMON_LINK_LIBS} )

set_target_properties( rocblas-bench PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
set_target_properties( rocblas_v3-bench PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")

add_dependencies( rocblas-bench rocblas-common )
add_dependencies( rocblas_v3-bench rocblas-common )

add_subdirectory ( ./perf_script )

rocm_install(TARGETS rocblas-bench COMPONENT benchmarks)
rocm_install(TARGETS rocblas_v3-bench COMPONENT benchmarks)
