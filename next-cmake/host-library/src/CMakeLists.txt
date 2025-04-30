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

set(_CMAKE_CURRENT_SOURCE_DIR ${TEMP_HOST_SOURCE_DIR}/src)

# rocblas L3 that use tensile but can use source gemm as fallback
set(rocblas_blas3_source
    # trsm first as currently slowest
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas3/rocblas_trsm.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas3/rocblas_trsm_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas3/rocblas_trsm_strided_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas3/rocblas_trsm_kernels.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas3/rocblas_trsm_batched_kernels.cpp
    #
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas3/rocblas_hemm.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas3/rocblas_hemm_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas3/rocblas_hemm_strided_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas3/rocblas_herk.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas3/rocblas_herk_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas3/rocblas_herk_strided_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas3/rocblas_her2k.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas3/rocblas_her2k_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas3/rocblas_her2k_strided_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas3/rocblas_herkx.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas3/rocblas_herkx_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas3/rocblas_herkx_strided_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas3/rocblas_symm.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas3/rocblas_symm_hemm_kernels.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas3/rocblas_symm_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas3/rocblas_symm_strided_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas3/rocblas_syrk.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas3/rocblas_syrk_herk_kernels.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas3/rocblas_syrk_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas3/rocblas_syrk_strided_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas3/rocblas_syr2k.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas3/rocblas_syr2k_her2k_kernels.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas3/rocblas_syr2k_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas3/rocblas_syr2k_strided_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas3/rocblas_gemm.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas3/rocblas_gemm_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas3/rocblas_gemm_strided_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas3/Tensile/gemm_templates.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas3/rocblas_syrkx.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas3/rocblas_syrkx_herkx_kernels.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas3/rocblas_syrkx_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas3/rocblas_syrkx_strided_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas3/rocblas_trmm.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas3/rocblas_trmm_kernels.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas3/rocblas_trmm_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas3/rocblas_trmm_strided_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas3/rocblas_trtri.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas3/rocblas_trtri_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas3/rocblas_trtri_strided_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas3/rocblas_trtri_kernels.cpp
)

set(rocblas_blas3_source_no_tensile
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas3/rocblas_dgmm.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas3/rocblas_dgmm_kernels.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas3/rocblas_dgmm_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas3/rocblas_dgmm_strided_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas3/rocblas_geam.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas3/rocblas_geam_kernels.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas3/rocblas_geam_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas3/rocblas_geam_strided_batched.cpp
)

set(rocblas_blas2_source
    # trsv first as slower
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_trsv.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_trsv_kernels.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_trsv_strided_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_trsv_batched.cpp
    #
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_gemv.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_gemv_kernels.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_gemv_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_gemv_strided_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_tpmv.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_tpmv_kernels.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_tpmv_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_tpmv_strided_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_gbmv.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_gbmv_kernels.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_gbmv_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_gbmv_strided_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_tbsv.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_tbsv_kernels.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_tbsv_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_tbsv_strided_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_trmv.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_trmv_kernels.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_trmv_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_trmv_strided_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_ger.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_ger_kernels.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_ger_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_ger_strided_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_hbmv.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_hbmv_kernels.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_hbmv_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_hbmv_strided_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_hemv.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_hemv_symv_kernels.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_hemv_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_hemv_strided_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_her.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_her_kernels.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_her_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_her_strided_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_her2.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_her2_kernels.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_her2_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_her2_strided_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_hpmv.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_hpmv_kernels.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_hpmv_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_hpmv_strided_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_hpr.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_hpr_kernels.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_hpr_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_hpr_strided_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_hpr2.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_hpr2_kernels.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_hpr2_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_hpr2_strided_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_spr.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_spr_kernels.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_spr_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_spr_strided_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_spr2.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_spr2_kernels.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_spr2_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_spr2_strided_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_syr.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_syr_kernels.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_syr_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_syr_strided_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_syr2.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_syr2_kernels.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_syr2_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_syr2_strided_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_tbmv.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_tbmv_kernels.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_tbmv_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_tbmv_strided_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_tpsv.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_tpsv_kernels.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_tpsv_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_tpsv_strided_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_sbmv.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_sbmv_kernels.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_sbmv_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_sbmv_strided_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_spmv.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_spmv_kernels.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_spmv_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_spmv_strided_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_symv.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_symv_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/rocblas_symv_strided_batched.cpp
)

set(rocblas_blas1_source
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas1/rocblas_iamax_iamin_kernels.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas1/rocblas_iamin.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas1/rocblas_iamin_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas1/rocblas_iamin_strided_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas1/rocblas_iamax.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas1/rocblas_iamax_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas1/rocblas_iamax_strided_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas1/rocblas_asum.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas1/rocblas_asum_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas1/rocblas_asum_strided_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas1/rocblas_axpy.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas1/rocblas_axpy_kernels.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas1/rocblas_axpy_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas1/rocblas_axpy_strided_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas1/rocblas_copy.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas1/rocblas_copy_kernels.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas1/rocblas_copy_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas1/rocblas_copy_strided_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas1/rocblas_dot.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas1/rocblas_dot_kernels.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas1/rocblas_dot_strided_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas1/rocblas_dot_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas1/rocblas_nrm2.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas1/rocblas_nrm2_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas1/rocblas_nrm2_strided_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas1/rocblas_asum_nrm2_kernels.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas1/rocblas_rot.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas1/rocblas_rot_kernels.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas1/rocblas_rot_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas1/rocblas_rot_strided_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas1/rocblas_rotg.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas1/rocblas_rotg_kernels.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas1/rocblas_rotg_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas1/rocblas_rotg_strided_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas1/rocblas_rotm.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas1/rocblas_rotm_kernels.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas1/rocblas_rotm_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas1/rocblas_rotm_strided_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas1/rocblas_rotmg.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas1/rocblas_rotmg_kernels.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas1/rocblas_rotmg_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas1/rocblas_rotmg_strided_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas1/rocblas_scal.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas1/rocblas_scal_kernels.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas1/rocblas_scal_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas1/rocblas_scal_strided_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas1/rocblas_swap.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas1/rocblas_swap_kernels.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas1/rocblas_swap_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas1/rocblas_swap_strided_batched.cpp
)

set(rocblas_ex_source_no_tensile
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas_ex/rocblas_axpy_ex.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas_ex/rocblas_axpy_ex_kernels.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas_ex/rocblas_axpy_batched_ex.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas_ex/rocblas_axpy_strided_batched_ex.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas_ex/rocblas_dot_ex.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas_ex/rocblas_dot_ex_kernels.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas_ex/rocblas_dot_batched_ex.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas_ex/rocblas_dot_strided_batched_ex.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas_ex/rocblas_rot_ex.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas_ex/rocblas_rot_ex_kernels.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas_ex/rocblas_rot_batched_ex.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas_ex/rocblas_rot_strided_batched_ex.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas_ex/rocblas_scal_ex.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas_ex/rocblas_scal_ex_kernels.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas_ex/rocblas_scal_batched_ex.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas_ex/rocblas_scal_strided_batched_ex.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas_ex/rocblas_nrm2_ex.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas_ex/rocblas_nrm2_ex_kernels.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas_ex/rocblas_nrm2_batched_ex.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas_ex/rocblas_nrm2_strided_batched_ex.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas_ex/rocblas_geam_ex.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas_ex/rocblas_geam_ex_kernels.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas_ex/rocblas_gemmt.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas_ex/rocblas_gemmt_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas_ex/rocblas_gemmt_strided_batched.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas_ex/rocblas_gemmt_kernels.cpp

    ${_CMAKE_CURRENT_SOURCE_DIR}/# these require may use Tensile or source gemm
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas_ex/rocblas_gemm_ex.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas_ex/rocblas_gemm_batched_ex.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas_ex/rocblas_gemm_strided_batched_ex.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas_ex/rocblas_gemm_ex_kernels.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas_ex/rocblas_trsv_ex.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas_ex/rocblas_trsv_strided_batched_ex.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas_ex/rocblas_trsv_batched_ex.cpp
)

set(rocblas_auxiliary_source
    ${_CMAKE_CURRENT_SOURCE_DIR}/handle.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/logging.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/rocblas_auxiliary.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/buildinfo.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/rocblas_ostream.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/check_numerics_vector.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/check_numerics_matrix.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/utility.cpp
)

# rocblas L3 that use tensile but can use source gemm as fallback
set(rocblas64_blas3_source
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas3/rocblas_hemm_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas3/rocblas_hemm_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas3/rocblas_hemm_strided_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas3/rocblas_herk_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas3/rocblas_herk_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas3/rocblas_herk_strided_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas3/rocblas_her2k_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas3/rocblas_her2k_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas3/rocblas_her2k_strided_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas3/rocblas_herkx_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas3/rocblas_herkx_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas3/rocblas_herkx_strided_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas3/rocblas_symm_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas3/rocblas_symm_hemm_kernels_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas3/rocblas_symm_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas3/rocblas_symm_strided_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas3/rocblas_syrk_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas3/rocblas_syrk_herk_kernels_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas3/rocblas_syrk_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas3/rocblas_syrk_strided_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas3/rocblas_syr2k_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas3/rocblas_syr2k_her2k_kernels_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas3/rocblas_syr2k_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas3/rocblas_syr2k_strided_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas3/rocblas_gemm_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas3/rocblas_gemm_kernels_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas3/rocblas_gemm_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas3/rocblas_gemm_strided_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas3/rocblas_syrkx_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas3/rocblas_syrkx_herkx_kernels_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas3/rocblas_syrkx_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas3/rocblas_syrkx_strided_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas3/rocblas_trmm_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas3/rocblas_trmm_kernels_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas3/rocblas_trmm_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas3/rocblas_trmm_strided_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas3/rocblas_trsm_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas3/rocblas_trsm_kernels_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas3/rocblas_trsm_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas3/rocblas_trsm_strided_batched_64.cpp
    #     src64/blas3/rocblas_trtri.cpp
    #     src64/blas3/rocblas_trtri_batched.cpp
    #     src64/blas3/rocblas_trtri_strided_batched.cpp
)

set(rocblas64_blas3_source_no_tensile
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas3/rocblas_dgmm_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas3/rocblas_dgmm_kernels_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas3/rocblas_dgmm_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas3/rocblas_dgmm_strided_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas3/rocblas_geam_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas3/rocblas_geam_kernels_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas3/rocblas_geam_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas3/rocblas_geam_strided_batched_64.cpp
)

set(rocblas64_blas2_source
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/rocblas_gemv_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/rocblas_gemv_kernels_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/rocblas_gemv_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/rocblas_gemv_strided_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/rocblas_tpmv_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/rocblas_tpmv_kernels_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/rocblas_tpmv_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/rocblas_tpmv_strided_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/rocblas_gbmv_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/rocblas_gbmv_kernels_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/rocblas_gbmv_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/rocblas_gbmv_strided_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/rocblas_tbsv_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/rocblas_tbsv_kernels_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/rocblas_tbsv_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/rocblas_tbsv_strided_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/rocblas_trmv_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/rocblas_trmv_kernels_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/rocblas_trmv_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/rocblas_trmv_strided_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/rocblas_ger_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/rocblas_ger_kernels_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/rocblas_ger_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/rocblas_ger_strided_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/rocblas_hbmv_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/rocblas_hbmv_kernels_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/rocblas_hbmv_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/rocblas_hbmv_strided_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/rocblas_hemv_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/rocblas_hemv_symv_kernels_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/rocblas_hemv_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/rocblas_hemv_strided_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/rocblas_her_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/rocblas_her_kernels_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/rocblas_her_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/rocblas_her_strided_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/rocblas_her2_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/rocblas_her2_kernels_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/rocblas_her2_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/rocblas_her2_strided_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/rocblas_hpmv_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/rocblas_hpmv_kernels_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/rocblas_hpmv_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/rocblas_hpmv_strided_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/rocblas_hpr_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/rocblas_hpr_kernels_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/rocblas_hpr_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/rocblas_hpr_strided_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/rocblas_hpr2_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/rocblas_hpr2_kernels_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/rocblas_hpr2_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/rocblas_hpr2_strided_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/rocblas_spr_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/rocblas_spr_kernels_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/rocblas_spr_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/rocblas_spr_strided_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/rocblas_spr2_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/rocblas_spr2_kernels_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/rocblas_spr2_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/rocblas_spr2_strided_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/rocblas_syr2_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/rocblas_syr2_kernels_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/rocblas_syr2_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/rocblas_syr2_strided_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/rocblas_tbmv_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/rocblas_tbmv_kernels_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/rocblas_tbmv_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/rocblas_tbmv_strided_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/rocblas_syr_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/rocblas_syr_kernels_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/rocblas_syr_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/rocblas_syr_strided_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/rocblas_tpsv_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/rocblas_tpsv_kernels_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/rocblas_tpsv_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/rocblas_tpsv_strided_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/rocblas_sbmv_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/rocblas_sbmv_kernels_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/rocblas_sbmv_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/rocblas_sbmv_strided_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/rocblas_spmv_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/rocblas_spmv_kernels_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/rocblas_spmv_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/rocblas_spmv_strided_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/rocblas_symv_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/rocblas_symv_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/rocblas_symv_strided_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/rocblas_trsv_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/rocblas_trsv_kernels_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/rocblas_trsv_strided_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/rocblas_trsv_batched_64.cpp
)

set(rocblas64_blas1_source
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas1/rocblas_iamax_iamin_kernels_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas1/rocblas_iamin_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas1/rocblas_iamin_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas1/rocblas_iamin_strided_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas1/rocblas_iamax_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas1/rocblas_iamax_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas1/rocblas_iamax_strided_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas1/rocblas_axpy_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas1/rocblas_axpy_kernels_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas1/rocblas_axpy_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas1/rocblas_axpy_strided_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas1/rocblas_copy_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas1/rocblas_copy_kernels_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas1/rocblas_copy_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas1/rocblas_copy_strided_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas1/rocblas_asum_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas1/rocblas_asum_nrm2_kernels_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas1/rocblas_asum_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas1/rocblas_asum_strided_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas1/rocblas_dot_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas1/rocblas_dot_kernels_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas1/rocblas_dot_strided_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas1/rocblas_dot_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas1/rocblas_nrm2_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas1/rocblas_nrm2_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas1/rocblas_nrm2_strided_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas1/rocblas_rot_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas1/rocblas_rot_kernels_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas1/rocblas_rot_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas1/rocblas_rot_strided_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas1/rocblas_rotg_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas1/rocblas_rotg_kernels_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas1/rocblas_rotg_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas1/rocblas_rotg_strided_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas1/rocblas_rotm_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas1/rocblas_rotm_kernels_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas1/rocblas_rotm_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas1/rocblas_rotm_strided_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas1/rocblas_rotmg_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas1/rocblas_rotmg_kernels_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas1/rocblas_rotmg_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas1/rocblas_rotmg_strided_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas1/rocblas_scal_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas1/rocblas_scal_kernels_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas1/rocblas_scal_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas1/rocblas_scal_strided_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas1/rocblas_swap_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas1/rocblas_swap_kernels_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas1/rocblas_swap_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas1/rocblas_swap_strided_batched_64.cpp
)

set(rocblas64_ex_source_no_tensile
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas_ex/rocblas_axpy_ex_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas_ex/rocblas_axpy_ex_kernels_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas_ex/rocblas_axpy_batched_ex_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas_ex/rocblas_axpy_strided_batched_ex_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas_ex/rocblas_dot_ex_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas_ex/rocblas_dot_ex_kernels_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas_ex/rocblas_dot_batched_ex_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas_ex/rocblas_dot_strided_batched_ex_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas_ex/rocblas_rot_ex_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas_ex/rocblas_rot_ex_kernels_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas_ex/rocblas_rot_batched_ex_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas_ex/rocblas_rot_strided_batched_ex_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas_ex/rocblas_scal_ex_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas_ex/rocblas_scal_ex_kernels_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas_ex/rocblas_scal_batched_ex_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas_ex/rocblas_scal_strided_batched_ex_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas_ex/rocblas_nrm2_ex_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas_ex/rocblas_nrm2_ex_kernels_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas_ex/rocblas_nrm2_batched_ex_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas_ex/rocblas_nrm2_strided_batched_ex_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas_ex/rocblas_gemmt_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas_ex/rocblas_gemmt_kernels_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas_ex/rocblas_gemmt_batched_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas_ex/rocblas_gemmt_strided_batched_64.cpp
    #     src64/blas_ex/rocblas_geam_ex.cpp
    #     src64/blas_ex/rocblas_geam_ex_kernels.cpp
)

set(rocblas64_ex_source
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas_ex/rocblas_gemm_ex_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas_ex/rocblas_gemm_batched_ex_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas_ex/rocblas_gemm_strided_batched_ex_64.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/src64/blas_ex/rocblas_gemm_ex_kernels_64.cpp
)


add_library(rocblas64-sources OBJECT)
target_sources(rocblas64-sources
    PRIVATE
        ${rocblas64_blas3_source}
        ${rocblas64_blas3_source_no_tensile}
        ${rocblas64_blas2_source}
        ${rocblas64_blas1_source}
        ${rocblas64_ex_source}
        ${rocblas64_ex_source_no_tensile}
)
target_include_directories(rocblas64-sources
    PRIVATE
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/../include>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/../include/internal>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/include/internal>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/src64>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../include/rocblas/internal>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../include/rocblas>  # TODO: Required?
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_compile_definitions(rocblas64-sources PRIVATE
    # TODO: Create a helper OBJECT library rocblas-common
    ROCBLAS_INTERNAL_ILP64  # <- this is the important one
    ROCM_USE_FLOAT16
    ROCBLAS_INTERNAL_API
    ROCBLAS_BETA_FEATURES_API
    rocblas_EXPORTS # TODO: needed?
)
target_link_libraries(rocblas64-sources PRIVATE hip::host hip::device)

# get_target_property(sources_tmp rocblas64-sources INTERFACE_SOURCES)

target_sources(rocblas
    PRIVATE
        ${rocblas_blas3_source}
        ${rocblas_blas3_source_no_tensile}
        ${rocblas_blas2_source}
        ${rocblas_blas1_source}
        # ${rocblas_ex_source}  # TODO: This doesn't exist?
        ${rocblas_ex_source_no_tensile}
        ${rocblas_auxiliary_source}
        # src64
)

target_link_libraries(rocblas PRIVATE rocblas64-sources)
