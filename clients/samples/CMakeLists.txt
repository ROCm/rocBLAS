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

set( rocblas_samples_common
      ../common/singletons.cpp
      ../common/utility.cpp
      ../common/rocblas_random.cpp
    )

# C example
add_executable( example-c-dgeam example_c_dgeam.c )
target_compile_definitions( example-c-dgeam PRIVATE __HIP_PLATFORM_HCC__ )
target_compile_options( example-c-dgeam PRIVATE -std=c11 )

# C++ examples
add_executable( example-sscal example_sscal.cpp ${rocblas_samples_common} )
add_executable( example-scal-template example_scal_template.cpp ${rocblas_samples_common} )
add_executable( example-solver-rocblas example_solver_rocblas.cpp ${rocblas_samples_common} )
add_executable( example-hip-complex-her2 example_hip_complex_her2.cpp )

if ( BUILD_FORTRAN_CLIENTS )
  # Fortran examples
  add_executable( example-fortran-axpy example_fortran_axpy.f90 $<TARGET_OBJECTS:rocblas_fortran>)
  add_executable( example-fortran-scal example_fortran_scal.f90 $<TARGET_OBJECTS:rocblas_fortran>)
  add_executable( example-fortran-gemv example_fortran_gemv.f90 $<TARGET_OBJECTS:rocblas_fortran>)
  set( sample_list_fortran example-fortran-axpy example-fortran-scal example-fortran-gemv )
endif( )

if( BUILD_WITH_TENSILE )
  add_executable( example-sgemm example_sgemm.cpp ${rocblas_samples_common} )
  add_executable( example-sgemm-strided-batched example_sgemm_strided_batched.cpp ${rocblas_samples_common} )
  add_executable( example-gemm-ext2 example_gemm_ext2.cpp ${rocblas_samples_common} )
  set( sample_list_tensile example-sgemm example-sgemm-strided-batched example-gemm-ext2 )
else( )
  add_executable( example-sgemm example_sgemm.cpp ${rocblas_samples_common} )
  add_executable( example-sgemm-strided-batched example_sgemm_strided_batched.cpp ${rocblas_samples_common} )
  set( sample_list_tensile example-sgemm example-sgemm-strided-batched )
endif( )

set( sample_list_c example-c-dgeam )
set( sample_list_base example-sscal example-scal-template example-solver-rocblas example-hip-complex-her2 )

set( sample_list_all ${sample_list_base} ${sample_list_tensile} ${sample_list_fortran} ${sample_list_c} )
set( sample_list_hip_device ${sample_list_base} ${sample_list_tensile} )

foreach( exe ${sample_list_fortran} )
  target_link_libraries( ${exe} PRIVATE rocblas_fortran_client )
endforeach( )

foreach( exe ${sample_list_all} )
  target_link_libraries( ${exe} PRIVATE roc::rocblas )

  set_target_properties( ${exe} PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging"
  )

  if (NOT ${exe} STREQUAL "example-c-dgeam")
    target_compile_definitions( ${exe} PRIVATE ROCM_USE_FLOAT16 )
  endif()

  target_include_directories( ${exe}
    PRIVATE
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../library/include>
      )

  target_include_directories( ${exe}
    SYSTEM PRIVATE
      $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
      )

  if( CUDA_FOUND )
    target_include_directories( ${exe}
      PRIVATE
        $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>
      )
    target_compile_definitions( ${exe} PRIVATE __HIP_PLATFORM_NVCC__ )
    target_link_libraries( ${exe} PRIVATE ${CUDA_LIBRARIES} )
  else( )
    # auto set in hip_common.h
    #target_compile_definitions( ${exe} PRIVATE __HIP_PLATFORM_HCC__ )
  endif( )

  if( CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # GCC or hip-clang needs specific flags to turn on f16c intrinsics
    target_compile_options( ${exe} PRIVATE -mf16c )
    target_compile_definitions( ${exe} PRIVATE ROCBLAS_INTERNAL_API )
  endif( )
  rocm_install(TARGETS ${exe} COMPONENT samples)
endforeach( )

foreach( exe ${sample_list_hip_device} )
  if( NOT CUDA_FOUND )
    target_link_libraries( ${exe} PRIVATE hip::device )
  endif()
endforeach( )
