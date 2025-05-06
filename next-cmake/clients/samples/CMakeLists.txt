# ########################################################################
# Copyright (C) 2025 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ########################################################################

set(TEMP_CLIENT_SAMPLES_DIR ${TEMP_CLIENTS_SOURCE_DIR}/samples)
set(_CMAKE_CURRENT_SOURCE_DIR ${TEMP_CLIENT_SAMPLES_DIR})

add_library(sample-helper INTERFACE)

if(ROCBLAS_ENABLE_BLIS)
    target_link_libraries(sample-helper INTERFACE BLIS::BLIS)
endif()
target_link_libraries(sample-helper
    INTERFACE
        hip::host
        hip::device
        roc::rocblas
)

target_compile_features(sample-helper INTERFACE cxx_std_14)

if( CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_definitions(sample-helper INTERFACE ROCBLAS_INTERNAL_API )
endif( )
target_compile_definitions(sample-helper INTERFACE
    ROCM_USE_FLOAT16
)

target_sources(sample-helper
    INTERFACE
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/../common/client_utility.cpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/../common/singletons.cpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/../common/rocblas_random.cpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/../common/host_alloc.cpp>
)

target_include_directories(sample-helper
    INTERFACE
        # $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/../../library/include/internal>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/../include>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/../common>
)

set_target_properties( ${exe} PROPERTIES
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging"
)

# Unused examples:
#    example_gemm_strided_batched_ex_xdl_math_op.cpp
#    example_openmp.cpp

# C examples
# add_subdirectory(example_c_dgeam)
# C++ examples
add_subdirectory(example_gemv_graph_capture)
add_subdirectory(example_header_check)
add_subdirectory(example_hip_complex_her2)
add_subdirectory(example_scal_multiple_strided_batch)
add_subdirectory(example_scal_template)
add_subdirectory(example_solver_rocblas)
add_subdirectory(example_sscal)
add_subdirectory(example_user_driven_tuning)
# L3 examples with or with tensile or source gemm builds
add_subdirectory(example_sgemm)
add_subdirectory(example_sgemm_multiple_strided_batch)
add_subdirectory(example_sgemm_strided_batched)

# if ( BUILD_FORTRAN_CLIENTS )
#   # Fortran examples
#   add_executable( rocblas-example-fortran-axpy example_fortran_axpy.f90 $<TARGET_OBJECTS:rocblas_fortran>)
#   add_executable( rocblas-example-fortran-scal example_fortran_scal.f90 $<TARGET_OBJECTS:rocblas_fortran>)
#   add_executable( rocblas-example-fortran-gemv example_fortran_gemv.f90 $<TARGET_OBJECTS:rocblas_fortran>)

#   set( sample_list_fortran rocblas-example-fortran-axpy rocblas-example-fortran-scal rocblas-example-fortran-gemv )
# endif()
