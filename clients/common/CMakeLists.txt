# ########################################################################
# Copyright (C) 2018-2024 Advanced Micro Devices, Inc. All rights reserved.
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


if( BUILD_WITH_TENSILE )
  set(rocblas_tensile_common_source

  )
endif()

set(rocblas_no_tensile_common_source
    # general
    gtest_helpers.cpp
    # rocblas_gtest_main.cpp
    # rocblas_test.cpp
    # general_gtest.cpp
    # set_get_pointer_mode_gtest.cpp
    # set_get_atomics_mode_gtest.cpp
    # logging_mode_gtest.cpp
    # ostream_threadsafety_gtest.cpp
    # set_get_vector_gtest.cpp
    # set_get_matrix_gtest.cpp
    # blas1
    blas1/common_asum.cpp
    blas1/common_axpy.cpp
    blas1/common_copy.cpp
    blas1/common_dot.cpp
    blas1/common_iamax_iamin.cpp
    blas1/common_nrm2.cpp
    blas1/common_rot.cpp
    blas1/common_scal.cpp
    blas1/common_swap.cpp
    # blas1_ex
    blas_ex/common_axpy_ex.cpp
    blas_ex/common_dot_ex.cpp
    blas_ex/common_nrm2_ex.cpp
    blas_ex/common_rot_ex.cpp
    blas_ex/common_scal_ex.cpp
    # blas2
    # blas2/trsv_gtest.cpp
    # blas2/gbmv_gtest.cpp
    # blas2/gemv_gtest.cpp
    # blas2/hbmv_gtest.cpp
    # blas2/hemv_gtest.cpp
    # blas2/her_gtest.cpp
    # blas2/her2_gtest.cpp
    # blas2/hpmv_gtest.cpp
    # blas2/hpr_gtest.cpp
    # blas2/hpr2_gtest.cpp
    # blas2/trmv_gtest.cpp
    # blas2/tpmv_gtest.cpp
    # blas2/tbmv_gtest.cpp
    # blas2/tbsv_gtest.cpp
    # blas2/tpsv_gtest.cpp
    # blas2/ger_gtest.cpp
    # blas2/geru_gtest.cpp
    # blas2/gerc_gtest.cpp
    # blas2/spr_gtest.cpp
    # blas2/spr2_gtest.cpp
    # blas2/syr_gtest.cpp
    # blas2/syr2_gtest.cpp
    # blas2/sbmv_gtest.cpp
    # blas2/spmv_gtest.cpp
    # blas2/symv_gtest.cpp
    # blas3 may use tensile or source gemm
    # blas3/gemm_gtest.cpp
    # blas_ex/gemm_ex_gtest.cpp
    # blas_ex/gemm_ex3_gtest.cpp
    # blas3/symm_gtest.cpp
    # blas3/hemm_gtest.cpp
    # blas3/trsm_gtest.cpp
    # blas3/trtri_gtest.cpp
    # blas3/trmm_gtest.cpp
    # blas3/syrk_gtest.cpp
    # blas3/syrkx_gtest.cpp
    # blas3/syr2k_gtest.cpp
    # blas3/herk_gtest.cpp
    # blas3/herkx_gtest.cpp
    # blas3/her2k_gtest.cpp
    # blas3/dgmm_gtest.cpp
    # blas3/geam_gtest.cpp
    # blas_ex/gemmt_gtest.cpp
    # blas_ex/geam_ex_gtest.cpp
  )

add_library(rocblas_clients_common OBJECT ${rocblas_no_tensile_common_source})

# Internal header includes
target_include_directories( rocblas_clients_common
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include/blas1>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include/blas2>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include/blas3>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include/blas_ex>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/blas_ex>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../library/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../library/src/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../library/src>
)

# External header includes included as system files
target_include_directories( rocblas_clients_common
  SYSTEM PRIVATE
    $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${BLAS_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${BLIS_INCLUDE_DIR}> # may be blank if not used
    $<BUILD_INTERFACE:${GTEST_INCLUDE_DIRS}>
)

target_include_directories( rocblas_clients_common
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/blas1>
)

target_compile_definitions( rocblas_clients_common PRIVATE ROCM_USE_FLOAT16 ROCBLAS_INTERNAL_API ROCBLAS_NO_DEPRECATED_WARNINGS )
target_compile_definitions( rocblas_clients_common PRIVATE ${TENSILE_DEFINES} GOOGLE_TEST)
if ( NOT BUILD_FORTRAN_CLIENTS )
  target_compile_definitions( rocblas_clients_common PRIVATE CLIENTS_NO_FORTRAN )
endif()

# target is not linked as OBJECT library but we want any interface flags from dependencies
target_link_libraries( rocblas_clients_common PRIVATE roc::rocblas hip::host hip::device ${GTEST_BOTH_LIBRARIES} )

