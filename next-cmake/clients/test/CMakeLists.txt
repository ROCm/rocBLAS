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

set(_CMAKE_CURRENT_SOURCE_DIR ${TEMP_CLIENTS_SOURCE_DIR}/gtest)


# TODO: Use if ROCBLAS_BUILD_WITH_TENSILE=ON
set(rocblas_tensile_test_source
    # Keep multiheaded_gtest.cpp first as we want to allow it to create the first TensileHost
    # Current GTESTs are run in the linking order as added with global variables
    ${_CMAKE_CURRENT_SOURCE_DIR}/multiheaded_gtest.cpp
    # use of tensile based functions (gemm)
    ${_CMAKE_CURRENT_SOURCE_DIR}/atomics_mode_gtest.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/get_solutions_gtest.cpp

)

set(rocblas_no_tensile_test_source
    # general
    ${_CMAKE_CURRENT_SOURCE_DIR}/rocblas_gtest_main.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/rocblas_test.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/general_gtest.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/set_get_pointer_mode_gtest.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/set_get_atomics_mode_gtest.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/logging_mode_gtest.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/ostream_threadsafety_gtest.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/set_get_vector_gtest.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/set_get_matrix_gtest.cpp
    # blas1
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas1/asum_gtest.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas1/axpy_gtest.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas1/copy_gtest.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas1/dot_gtest.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas1/iamaxmin_gtest.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas1/nrm2_gtest.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas1/rot_gtest.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas1/scal_gtest.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas1/swap_gtest.cpp
    # blas1_ex
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas_ex/axpy_ex_gtest.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas_ex/dot_ex_gtest.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas_ex/nrm2_ex_gtest.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas_ex/rot_ex_gtest.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas_ex/scal_ex_gtest.cpp
    # blas2
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/trsv_gtest.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/gbmv_gtest.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/gemv_gtest.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/hbmv_gtest.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/hemv_gtest.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/her_gtest.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/her2_gtest.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/hpmv_gtest.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/hpr_gtest.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/hpr2_gtest.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/trmv_gtest.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/tpmv_gtest.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/tbmv_gtest.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/tbsv_gtest.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/tpsv_gtest.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/ger_gtest.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/geru_gtest.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/gerc_gtest.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/spr_gtest.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/spr2_gtest.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/syr_gtest.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/syr2_gtest.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/sbmv_gtest.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/spmv_gtest.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/symv_gtest.cpp
    # blas3 may use tensile or source gemm
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas3/gemm_gtest.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas_ex/gemm_ex_gtest.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas3/symm_gtest.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas3/hemm_gtest.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas3/trsm_gtest.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas3/trtri_gtest.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas3/trmm_gtest.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas3/syrk_gtest.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas3/syrkx_gtest.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas3/syr2k_gtest.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas3/herk_gtest.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas3/herkx_gtest.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas3/her2k_gtest.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas3/dgmm_gtest.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas3/geam_gtest.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas_ex/gemmt_gtest.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas_ex/geam_ex_gtest.cpp
  )

add_executable(rocblas-test)

target_sources(rocblas-test
    PRIVATE
        ${rocblas_tensile_test_source}
        ${rocblas_no_tensile_test_source}
)

target_link_libraries(rocblas-test
    PRIVATE
        rocblas-clients-common
)

set_target_properties(rocblas-test
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging"
)

# TODO: Have not implemented rocblas testing custom commands
