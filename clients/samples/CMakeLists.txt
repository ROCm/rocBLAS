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

set( rocblas_samples_common
      ../common/singletons.cpp
      ../common/utility.cpp
      ../common/rocblas_random.cpp
    )

# C example
add_executable( rocblas-example-c-dgeam example_c_dgeam.c )
target_compile_definitions( rocblas-example-c-dgeam PRIVATE __HIP_PLATFORM_HCC__ )
target_compile_options( rocblas-example-c-dgeam PRIVATE -std=c11 )

# C++ examples
add_executable( rocblas-example-sscal example_sscal.cpp ${rocblas_samples_common} )
add_executable( rocblas-example-scal-template example_scal_template.cpp ${rocblas_samples_common} )
add_executable( rocblas-example-solver example_solver_rocblas.cpp ${rocblas_samples_common} )
add_executable( rocblas-example-hip-complex-her2 example_hip_complex_her2.cpp )
add_executable( rocblas-example-gemv-graph-capture example_gemv_graph_capture.cpp )

if ( BUILD_FORTRAN_CLIENTS )
  # Fortran examples
  add_executable( rocblas-example-fortran-axpy example_fortran_axpy.f90 $<TARGET_OBJECTS:rocblas_fortran>)
  add_executable( rocblas-example-fortran-scal example_fortran_scal.f90 $<TARGET_OBJECTS:rocblas_fortran>)
  add_executable( rocblas-example-fortran-gemv example_fortran_gemv.f90 $<TARGET_OBJECTS:rocblas_fortran>)
  add_executable( rocblas-example-fortran-trmm_v3 example_fortran_trmm_v3.F90 $<TARGET_OBJECTS:rocblas_fortran>)
  set( sample_list_fortran rocblas-example-fortran-axpy rocblas-example-fortran-scal rocblas-example-fortran-gemv rocblas-example-fortran-trmm_v3 )
endif( )

if( BUILD_WITH_TENSILE )
  add_executable( rocblas-example-user-driven-tuning example_user_driven_tuning.cpp ${rocblas_samples_common} )
  add_executable( rocblas-example-sgemm example_sgemm.cpp ${rocblas_samples_common} )
  add_executable( rocblas-example-strmm_v3 example_strmm_v3.cpp ${rocblas_samples_common} )
  add_executable( rocblas-example-sgemm-strided-batched example_sgemm_strided_batched.cpp ${rocblas_samples_common} )
  add_executable( rocblas-example-gemm-ext2 example_gemm_ext2.cpp ${rocblas_samples_common} )
  set( sample_list_tensile rocblas-example-user-driven-tuning rocblas-example-sgemm rocblas-example-strmm_v3 rocblas-example-sgemm-strided-batched rocblas-example-gemm-ext2 )
else( )
  add_executable( rocblas-example-sgemm example_sgemm.cpp ${rocblas_samples_common} )
  add_executable( rocblas-example-strmm_v3 example_strmm_v3.cpp ${rocblas_samples_common} )
  add_executable( rocblas-example-sgemm-strided-batched example_sgemm_strided_batched.cpp ${rocblas_samples_common} )
  set( sample_list_tensile rocblas-example-sgemm rocblas-example-strmm_v3 rocblas-example-sgemm-strided-batched )
endif( )

set( sample_list_c rocblas-example-c-dgeam )
set( sample_list_base rocblas-example-sscal rocblas-example-scal-template rocblas-example-solver rocblas-example-hip-complex-her2 rocblas-example-gemv-graph-capture)

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

  if (NOT ${exe} STREQUAL "rocblas-example-c-dgeam")
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

target_compile_definitions( rocblas-example-strmm_v3 PRIVATE ROCBLAS_V3)
if ( BUILD_FORTRAN_CLIENTS )
  target_compile_definitions( rocblas-example-fortran-trmm_v3 PRIVATE ROCBLAS_V3)
endif( )
