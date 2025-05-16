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

set(_CMAKE_CURRENT_SOURCE_DIR ${TEMP_CLIENTS_SOURCE_DIR}/common/src)

target_sources(
  rocblas-clients-common
  PRIVATE # general
          ${_CMAKE_CURRENT_SOURCE_DIR}/../singletons.cpp
          ${_CMAKE_CURRENT_SOURCE_DIR}/../client_omp.cpp
          ${_CMAKE_CURRENT_SOURCE_DIR}/../client_utility.cpp
          ${_CMAKE_CURRENT_SOURCE_DIR}/../frequency_monitor.cpp
          ${_CMAKE_CURRENT_SOURCE_DIR}/../rocblas_arguments.cpp
          ${_CMAKE_CURRENT_SOURCE_DIR}/../argument_model.cpp
          ${_CMAKE_CURRENT_SOURCE_DIR}/../rocblas_random.cpp
          ${_CMAKE_CURRENT_SOURCE_DIR}/../rocblas_parse_data.cpp
          ${_CMAKE_CURRENT_SOURCE_DIR}/../host_alloc.cpp
          ${_CMAKE_CURRENT_SOURCE_DIR}/../gtest_helpers.cpp
          ${_CMAKE_CURRENT_SOURCE_DIR}/../rocblas_matrix.cpp
          ${_CMAKE_CURRENT_SOURCE_DIR}/../cblas_interface.cpp
          # blas1
          ${_CMAKE_CURRENT_SOURCE_DIR}/../blas1/common_asum.cpp
          ${_CMAKE_CURRENT_SOURCE_DIR}/../blas1/common_axpy.cpp
          ${_CMAKE_CURRENT_SOURCE_DIR}/../blas1/common_copy.cpp
          ${_CMAKE_CURRENT_SOURCE_DIR}/../blas1/common_dot.cpp
          ${_CMAKE_CURRENT_SOURCE_DIR}/../blas1/common_iamax_iamin.cpp
          ${_CMAKE_CURRENT_SOURCE_DIR}/../blas1/common_nrm2.cpp
          ${_CMAKE_CURRENT_SOURCE_DIR}/../blas1/common_rot.cpp
          ${_CMAKE_CURRENT_SOURCE_DIR}/../blas1/common_scal.cpp
          ${_CMAKE_CURRENT_SOURCE_DIR}/../blas1/common_swap.cpp
          # blas1_ex
          ${_CMAKE_CURRENT_SOURCE_DIR}/../blas_ex/common_axpy_ex.cpp
          ${_CMAKE_CURRENT_SOURCE_DIR}/../blas_ex/common_dot_ex.cpp
          ${_CMAKE_CURRENT_SOURCE_DIR}/../blas_ex/common_nrm2_ex.cpp
          ${_CMAKE_CURRENT_SOURCE_DIR}/../blas_ex/common_rot_ex.cpp
          ${_CMAKE_CURRENT_SOURCE_DIR}/../blas_ex/common_scal_ex.cpp
          # blas2
          ${_CMAKE_CURRENT_SOURCE_DIR}/../blas2/common_trsv.cpp
          ${_CMAKE_CURRENT_SOURCE_DIR}/../blas2/common_gbmv.cpp
          ${_CMAKE_CURRENT_SOURCE_DIR}/../blas2/common_gemv.cpp
          ${_CMAKE_CURRENT_SOURCE_DIR}/../blas2/common_hbmv.cpp
          ${_CMAKE_CURRENT_SOURCE_DIR}/../blas2/common_hemv.cpp
          ${_CMAKE_CURRENT_SOURCE_DIR}/../blas2/common_her.cpp
          ${_CMAKE_CURRENT_SOURCE_DIR}/../blas2/common_her2.cpp
          ${_CMAKE_CURRENT_SOURCE_DIR}/../blas2/common_hpmv.cpp
          ${_CMAKE_CURRENT_SOURCE_DIR}/../blas2/common_hpr.cpp
          ${_CMAKE_CURRENT_SOURCE_DIR}/../blas2/common_hpr2.cpp
          ${_CMAKE_CURRENT_SOURCE_DIR}/../blas2/common_trmv.cpp
          ${_CMAKE_CURRENT_SOURCE_DIR}/../blas2/common_tpmv.cpp
          ${_CMAKE_CURRENT_SOURCE_DIR}/../blas2/common_tbmv.cpp
          ${_CMAKE_CURRENT_SOURCE_DIR}/../blas2/common_tbsv.cpp
          ${_CMAKE_CURRENT_SOURCE_DIR}/../blas2/common_tpsv.cpp
          ${_CMAKE_CURRENT_SOURCE_DIR}/../blas2/common_ger.cpp
          ${_CMAKE_CURRENT_SOURCE_DIR}/../blas2/common_spr.cpp
          ${_CMAKE_CURRENT_SOURCE_DIR}/../blas2/common_spr2.cpp
          ${_CMAKE_CURRENT_SOURCE_DIR}/../blas2/common_syr.cpp
          ${_CMAKE_CURRENT_SOURCE_DIR}/../blas2/common_syr2.cpp
          ${_CMAKE_CURRENT_SOURCE_DIR}/../blas2/common_sbmv.cpp
          ${_CMAKE_CURRENT_SOURCE_DIR}/../blas2/common_spmv.cpp
          ${_CMAKE_CURRENT_SOURCE_DIR}/../blas2/common_symv.cpp
          # blas3 may use tensile or source gemm
          ${_CMAKE_CURRENT_SOURCE_DIR}/../blas3/common_gemm.cpp
          ${_CMAKE_CURRENT_SOURCE_DIR}/../blas_ex/common_gemm_ex.cpp
          ${_CMAKE_CURRENT_SOURCE_DIR}/../blas_ex/common_trsm_ex.cpp
          ${_CMAKE_CURRENT_SOURCE_DIR}/../blas3/common_symm_hemm.cpp
          ${_CMAKE_CURRENT_SOURCE_DIR}/../blas3/common_trsm.cpp
          ${_CMAKE_CURRENT_SOURCE_DIR}/../blas3/common_trtri.cpp
          ${_CMAKE_CURRENT_SOURCE_DIR}/../blas3/common_trmm.cpp
          ${_CMAKE_CURRENT_SOURCE_DIR}/../blas3/common_syrk.cpp
          ${_CMAKE_CURRENT_SOURCE_DIR}/../blas3/common_syr2k.cpp # syrkx
          ${_CMAKE_CURRENT_SOURCE_DIR}/../blas3/common_herk.cpp
          ${_CMAKE_CURRENT_SOURCE_DIR}/../blas3/common_her2k.cpp # herkx
          ${_CMAKE_CURRENT_SOURCE_DIR}/../blas3/common_dgmm.cpp
          ${_CMAKE_CURRENT_SOURCE_DIR}/../blas3/common_geam.cpp
          # blas_ex
          ${_CMAKE_CURRENT_SOURCE_DIR}/../blas_ex/common_gemmt.cpp
          ${_CMAKE_CURRENT_SOURCE_DIR}/../blas_ex/common_geam_ex.cpp)
