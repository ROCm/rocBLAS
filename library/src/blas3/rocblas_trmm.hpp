/* ************************************************************************
 * Copyright 2019-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "Tensile/gemm.hpp"
#include "definitions.hpp"

template <typename TScal, typename TPtr, typename T_lda>
ROCBLAS_KERNEL void set_matrix_zero_if_alpha_zero_kernel(rocblas_int    m,
                                                         rocblas_int    n,
                                                         TScal          alpha_device_host,
                                                         rocblas_stride stride_alpha,
                                                         TPtr           Aa,
                                                         T_lda          offsetA,
                                                         T_lda          lda,
                                                         rocblas_stride strideA);

template <typename TScal, typename TPtr, typename T_lda>
rocblas_status set_matrix_zero_if_alpha_zero_template(rocblas_handle handle,
                                                      rocblas_int    m,
                                                      rocblas_int    n,
                                                      TScal          alpha,
                                                      rocblas_stride stride_alpha,
                                                      TPtr           A,
                                                      T_lda          offsetA,
                                                      T_lda          lda,
                                                      rocblas_stride strideA,
                                                      rocblas_int    batch_count);

template <int  NB,
          bool BATCHED,
          typename T,
          typename TScal,
          typename TConstPtr,
          typename TPtr,
          typename T_lda>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_trmm_template(rocblas_handle    handle,
                                   rocblas_side      side,
                                   rocblas_fill      uplo,
                                   rocblas_operation trans_a,
                                   rocblas_diagonal  diag,
                                   rocblas_int       m,
                                   rocblas_int       n,
                                   TScal*            alpha,
                                   rocblas_stride    stride_alpha,
                                   TConstPtr*        dA,
                                   T_lda             offset_a,
                                   T_lda             ldda,
                                   rocblas_stride    stride_a,
                                   TConstPtr*        dB,
                                   T_lda             offset_b,
                                   T_lda             lddb,
                                   rocblas_stride    stride_b,
                                   TPtr*             dC,
                                   T_lda             offset_c,
                                   T_lda             lddc,
                                   rocblas_stride    stride_c,
                                   rocblas_int       batch_count);
