/* ************************************************************************
 * Copyright 2020-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "handle.hpp"

template <typename TScal, typename TConstPtr, typename TPtr>
inline rocblas_status rocblas_syrk_arg_check(rocblas_handle    handle,
                                             rocblas_fill      uplo,
                                             rocblas_operation transA,
                                             rocblas_int       n,
                                             rocblas_int       k,
                                             const TScal*      alpha,
                                             TConstPtr         AP,
                                             rocblas_int       offsetA,
                                             rocblas_int       lda,
                                             rocblas_stride    strideA,
                                             const TScal*      beta,
                                             TPtr              CP,
                                             rocblas_int       offsetC,
                                             rocblas_int       ldc,
                                             rocblas_stride    strideC,
                                             rocblas_int       batch_count)
{
    if(uplo != rocblas_fill_lower && uplo != rocblas_fill_upper)
        return rocblas_status_invalid_value;
    if(transA == rocblas_operation_conjugate_transpose && is_complex<TScal>)
        return rocblas_status_invalid_value;

    if(n < 0 || k < 0 || batch_count < 0 || ldc < n || (transA == rocblas_operation_none && lda < n)
       || (transA != rocblas_operation_none && lda < k))
        return rocblas_status_invalid_size;
    if(!n || !batch_count)
        return rocblas_status_success;

    if((k > 0 && (!AP || !alpha)) || !CP || !beta)
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <typename TScal, typename TConstPtr, typename TPtr>
inline rocblas_status rocblas_herk_arg_check(rocblas_handle    handle,
                                             rocblas_fill      uplo,
                                             rocblas_operation transA,
                                             rocblas_int       n,
                                             rocblas_int       k,
                                             TScal             alpha,
                                             TConstPtr         AP,
                                             rocblas_int       offsetA,
                                             rocblas_int       lda,
                                             rocblas_stride    strideA,
                                             TScal             beta,
                                             TPtr              CP,
                                             rocblas_int       offsetC,
                                             rocblas_int       ldc,
                                             rocblas_stride    strideC,
                                             rocblas_int       batch_count)
{
    if(uplo != rocblas_fill_lower && uplo != rocblas_fill_upper)
        return rocblas_status_invalid_value;
    if(transA != rocblas_operation_none && transA != rocblas_operation_conjugate_transpose)
        return rocblas_status_invalid_value;

    if(n < 0 || k < 0 || batch_count < 0 || ldc < n || (transA == rocblas_operation_none && lda < n)
       || (transA != rocblas_operation_none && lda < k))
        return rocblas_status_invalid_size;
    if(!n || !batch_count)
        return rocblas_status_success;
    if((k > 0 && (!AP || !alpha)) || !CP || !beta)
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

/**
  *  TScal     is always: const T* (either host or device)
  *  TConstPtr is either: const T* OR const T* const*
  *  TPtr      is either:       T* OR       T* const*
  */
template <typename TScal, typename TConstPtr, typename TPtr>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_syrk_template(rocblas_handle    handle,
                                   rocblas_fill      uplo,
                                   rocblas_operation transA,
                                   rocblas_int       n,
                                   rocblas_int       k,
                                   TScal             alpha,
                                   TConstPtr         AP,
                                   rocblas_int       offsetA,
                                   rocblas_int       lda,
                                   rocblas_stride    strideA,
                                   TScal             beta,
                                   TPtr              CP,
                                   rocblas_int       offsetC,
                                   rocblas_int       ldc,
                                   rocblas_stride    strideC,
                                   rocblas_int       batch_count);

/**
  *  TScal     is always: const T* (either host or device)
  *  TConstPtr is either: const T* OR const T* const*
  *  TPtr      is either:       T* OR       T* const*
  */
template <typename TScal, typename TConstPtr, typename TPtr>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_herk_template(rocblas_handle    handle,
                                   rocblas_fill      uplo,
                                   rocblas_operation transA,
                                   rocblas_int       n,
                                   rocblas_int       k,
                                   TScal             alpha,
                                   TConstPtr         AP,
                                   rocblas_int       offsetA,
                                   rocblas_int       lda,
                                   rocblas_stride    strideA,
                                   TScal             beta,
                                   TPtr              CP,
                                   rocblas_int       offsetC,
                                   rocblas_int       ldc,
                                   rocblas_stride    strideC,
                                   rocblas_int       batch_count);
