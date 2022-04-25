/* ************************************************************************
 * Copyright 2019-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "Tensile/gemm.hpp"
#include "definitions.hpp"

template <rocblas_int DIM_X, rocblas_int DIM_Y, typename TScal, typename TPtr>
ROCBLAS_KERNEL(DIM_X* DIM_Y)
set_matrix_zero_if_alpha_zero_kernel(rocblas_int    m,
                                     rocblas_int    n,
                                     TScal          alpha_device_host,
                                     rocblas_stride stride_alpha,
                                     TPtr           Aa,
                                     rocblas_int    lda,
                                     rocblas_stride a_st_or_of);

template <typename TScal, typename TPtr>
rocblas_status set_matrix_zero_if_alpha_zero_template(rocblas_handle handle,
                                                      rocblas_int    m,
                                                      rocblas_int    n,
                                                      TScal          alpha,
                                                      rocblas_stride stride_alpha,
                                                      TPtr           A,
                                                      rocblas_int    lda,
                                                      rocblas_stride a_st_or_of,
                                                      rocblas_int    batch_count);

template <typename TScal, typename TPtr, typename TConstPtr>
rocblas_status rocblas_trmm_arg_check(rocblas_handle    handle,
                                      rocblas_side      side,
                                      rocblas_fill      uplo,
                                      rocblas_operation trans,
                                      rocblas_diagonal  diag,
                                      rocblas_int       m,
                                      rocblas_int       n,
                                      const TScal*      alpha,
                                      TConstPtr         a,
                                      rocblas_int       lda,
                                      TPtr              b,
                                      rocblas_int       ldb,
                                      rocblas_int       batch_count)
{
    if(side != rocblas_side_left && side != rocblas_side_right)
        return rocblas_status_invalid_value;

    if(uplo != rocblas_fill_lower && uplo != rocblas_fill_upper)
        return rocblas_status_invalid_value;

    if(trans != rocblas_operation_none && trans != rocblas_operation_transpose
       && trans != rocblas_operation_conjugate_transpose)
        return rocblas_status_invalid_value;

    if(diag != rocblas_diagonal_non_unit && diag != rocblas_diagonal_unit)
        return rocblas_status_invalid_value;

    if(batch_count < 0 || m < 0 || n < 0 || ldb < m || (side == rocblas_side_left && (lda < m))
       || (side != rocblas_side_left && (lda < n)))
        return rocblas_status_invalid_size;

    if(!m || !n || !batch_count)
        return rocblas_status_success;

    if(!b || !alpha || (handle->pointer_mode == rocblas_pointer_mode_host && *alpha != 0 && !a))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <typename TScal, typename TPtr, typename TConstPtr>
rocblas_status rocblas_trmm_outofplace_arg_check(rocblas_handle    handle,
                                                 rocblas_side      side,
                                                 rocblas_fill      uplo,
                                                 rocblas_operation trans,
                                                 rocblas_diagonal  diag,
                                                 rocblas_int       m,
                                                 rocblas_int       n,
                                                 const TScal*      alpha,
                                                 TConstPtr         a,
                                                 rocblas_int       lda,
                                                 TConstPtr         b,
                                                 rocblas_int       ldb,
                                                 TPtr              c,
                                                 rocblas_int       ldc,
                                                 rocblas_int       batch_count)
{
    if(side != rocblas_side_left && side != rocblas_side_right)
        return rocblas_status_invalid_value;

    if(uplo != rocblas_fill_lower && uplo != rocblas_fill_upper)
        return rocblas_status_invalid_value;

    if(trans != rocblas_operation_none && trans != rocblas_operation_transpose
       && trans != rocblas_operation_conjugate_transpose)
        return rocblas_status_invalid_value;

    if(diag != rocblas_diagonal_non_unit && diag != rocblas_diagonal_unit)
        return rocblas_status_invalid_value;

    if(batch_count < 0 || m < 0 || n < 0 || ldc < m || ldb < m
       || (side == rocblas_side_left && (lda < m)) || (side != rocblas_side_left && (lda < n)))
        return rocblas_status_invalid_size;

    if(!m || !n || !batch_count)
        return rocblas_status_success;

    if(!c || !alpha
       || (handle->pointer_mode == rocblas_pointer_mode_host && *alpha != 0 && (!a || !b)))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

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
                                              rocblas_stride    offset_a,
                                              rocblas_int       lda,
                                              rocblas_stride    stride_a,
                                              TConstPtr*        dB,
                                              rocblas_stride    offset_b,
                                              rocblas_int       ldb,
                                              rocblas_stride    stride_b,
                                              TPtr*             dC,
                                              rocblas_stride    offset_c,
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
    rocblas_internal_trmm_recursive_template(rocblas_handle    handle,
                                             rocblas_side      side,
                                             rocblas_fill      uplo,
                                             rocblas_operation trans_a,
                                             rocblas_diagonal  diag,
                                             rocblas_int       m,
                                             rocblas_int       n,
                                             TScal*            alpha,
                                             rocblas_stride    stride_alpha,
                                             TConstPtr*        dA,
                                             rocblas_stride    offset_a,
                                             rocblas_int       lda,
                                             rocblas_stride    stride_a,
                                             TPtr*             dB,
                                             rocblas_stride    offset_b,
                                             rocblas_int       ldb,
                                             rocblas_stride    stride_b,
                                             TPtr*             dC,
                                             rocblas_stride    offset_c,
                                             rocblas_int       ldc,
                                             rocblas_stride    stride_c,
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
                                   rocblas_stride    offset_a,
                                   rocblas_int       lda,
                                   rocblas_stride    stride_a,
                                   TConstPtr*        dB,
                                   rocblas_stride    offset_b,
                                   rocblas_int       ldb,
                                   rocblas_stride    stride_b,
                                   TPtr*             dC,
                                   rocblas_stride    offset_c,
                                   rocblas_int       lddc,
                                   rocblas_stride    stride_c,
                                   rocblas_int       batch_count);
