/* ************************************************************************
 * Copyright 2019-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "Tensile/gemm.hpp"
#include "definitions.hpp"

template <typename TScal, typename TPtr>
ROCBLAS_KERNEL void set_matrix_zero_if_alpha_zero_kernel(rocblas_int    m,
                                                         rocblas_int    n,
                                                         TScal          alpha_device_host,
                                                         rocblas_stride stride_alpha,
                                                         TPtr           Aa,
                                                         rocblas_int    offsetA,
                                                         rocblas_int    lda,
                                                         rocblas_stride strideA);

template <typename TScal, typename TPtr>
rocblas_status set_matrix_zero_if_alpha_zero_template(rocblas_handle handle,
                                                      rocblas_int    m,
                                                      rocblas_int    n,
                                                      TScal          alpha,
                                                      rocblas_stride stride_alpha,
                                                      TPtr           A,
                                                      rocblas_int    offsetA,
                                                      rocblas_int    lda,
                                                      rocblas_stride strideA,
                                                      rocblas_int    batch_count);

template <int  NB,
          bool BATCHED,
          bool CONJ,
          typename T,
          typename TScal,
          typename TConstPtr,
          typename TPtr>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_trmm_outofplace_template(rocblas_handle    handle,
                                              rocblas_side      side,
                                              rocblas_fill      uplo,
                                              rocblas_operation trans_a,
                                              rocblas_diagonal  diag,
                                              rocblas_int       m,
                                              rocblas_int       n,
                                              TScal*            alpha,
                                              rocblas_stride    stride_alpha,
                                              TConstPtr*        dA,
                                              rocblas_int       offset_a,
                                              rocblas_int       lda,
                                              rocblas_stride    stride_a,
                                              TConstPtr*        dB,
                                              rocblas_int       offset_b,
                                              rocblas_int       ldb,
                                              rocblas_stride    stride_b,
                                              TPtr*             dC,
                                              rocblas_int       offset_c,
                                              rocblas_int       lddc,
                                              rocblas_stride    stride_c,
                                              rocblas_int       batch_count);

template <int  STOPPING_NB,
          bool BATCHED,
          typename T,
          typename TScal,
          typename TConstPtr,
          typename TPtr>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_trmm_recursive_inplace_template(rocblas_handle    handle,
                                                     rocblas_side      side,
                                                     rocblas_fill      uplo,
                                                     rocblas_operation trans_a,
                                                     rocblas_diagonal  diag,
                                                     rocblas_int       m,
                                                     rocblas_int       n,
                                                     TScal*            alpha,
                                                     rocblas_stride    stride_alpha,
                                                     TConstPtr*        dA,
                                                     rocblas_int       offset_a,
                                                     rocblas_int       lda,
                                                     rocblas_stride    stride_a,
                                                     TPtr*             dB,
                                                     rocblas_int       offset_b,
                                                     rocblas_int       ldb,
                                                     rocblas_stride    stride_b,
                                                     rocblas_int       batch_count);

template <int NB, bool BATCHED, typename T, typename TScal, typename TConstPtr, typename TPtr>
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
                                   rocblas_int       offset_a,
                                   rocblas_int       lda,
                                   rocblas_stride    stride_a,
                                   TConstPtr*        dB,
                                   rocblas_int       offset_b,
                                   rocblas_int       ldb,
                                   rocblas_stride    stride_b,
                                   TPtr*             dC,
                                   rocblas_int       offset_c,
                                   rocblas_int       lddc,
                                   rocblas_stride    stride_c,
                                   rocblas_int       batch_count);
