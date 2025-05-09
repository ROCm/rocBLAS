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

set(_CMAKE_CURRENT_SOURCE_DIR ${TEMP_HOST_SOURCE_DIR}/src/src64)

target_sources(rocblas64
    PRIVATE
        "${_CMAKE_CURRENT_SOURCE_DIR}/blas1/rocblas_asum_nrm2_64.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/blas1/rocblas_asum_nrm2_kernels_64.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/blas1/rocblas_axpy_64.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/blas1/rocblas_copy_64.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/blas1/rocblas_dot_64.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/blas1/rocblas_iamax_iamin_64.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/blas1/rocblas_iamax_iamin_kernels_64.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/blas1/rocblas_nrm2_64.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/blas1/rocblas_rot_64.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/blas1/rocblas_rotg_64.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/blas1/rocblas_rotm_64.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/blas1/rocblas_rotmg_64.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/blas1/rocblas_scal_64.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/blas1/rocblas_swap_64.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_gbmv_64.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_gemv_64.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_ger_64.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_hbmv_64.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_hemv_symv_64.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_her2_64.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_her_64.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_hpmv_64.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_hpr2_64.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_hpr_64.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_sbmv_64.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_spmv_64.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_spr2_64.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_spr_64.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_syr2_64.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_syr_64.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_tbmv_64.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_tbsv_64.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_tpmv_64.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_tpsv_64.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_trmv_64.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_trsv_64.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/blas3/rocblas_dgmm_64.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/blas3/rocblas_geam_64.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/blas3/rocblas_gemm_64.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/blas3/rocblas_symm_hemm_64.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/blas3/rocblas_syr2k_her2k_64.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/blas3/rocblas_syrk_herk_64.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/blas3/rocblas_syrkx_herkx_64.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/blas3/rocblas_trmm_64.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/blas3/rocblas_trsm_64.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/blas_ex/rocblas_gemm_ex_64.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/blas_ex/rocblas_gemmt_64.hpp"
)

