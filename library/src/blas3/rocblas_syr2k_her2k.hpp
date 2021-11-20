/* ************************************************************************
 * Copyright 2020-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "handle.hpp"
#include "herk_scale_device.hpp"

template <bool        BATCHED,
          bool        TWOK,
          bool        HERM,
          bool        TRANS,
          rocblas_int DIM_XYT,
          typename TScal,
          typename TConstPtr,
          typename TPtr>
ROCBLAS_KERNEL __launch_bounds__(DIM_XYT* DIM_XYT) void syr2k_her2k_kernel(bool              upper,
                                                                           rocblas_operation trans,
                                                                           rocblas_int       n,
                                                                           rocblas_int       k,
                                                                           TScal alpha_host_device,
                                                                           TConstPtr      AP_array,
                                                                           rocblas_stride shift_a,
                                                                           rocblas_int    lda,
                                                                           rocblas_stride stride_a,
                                                                           TConstPtr      BP_array,
                                                                           rocblas_stride shift_b,
                                                                           rocblas_int    ldb,
                                                                           rocblas_stride stride_b,
                                                                           TPtr           CP_array,
                                                                           rocblas_stride shift_c,
                                                                           rocblas_int    ldc,
                                                                           rocblas_stride stride_c);

template <typename TScal, typename TConstPtr, typename TPtr>
inline rocblas_status rocblas_syr2k_arg_check(rocblas_handle    handle,
                                              rocblas_fill      uplo,
                                              rocblas_operation trans,
                                              rocblas_int       n,
                                              rocblas_int       k,
                                              TScal             alpha,
                                              TConstPtr         AP,
                                              rocblas_stride    offsetA,
                                              rocblas_int       lda,
                                              rocblas_stride    strideA,
                                              TConstPtr         BP,
                                              rocblas_stride    offsetB,
                                              rocblas_int       ldb,
                                              rocblas_stride    strideB,
                                              TScal             beta,
                                              TPtr              CP,
                                              rocblas_stride    offsetC,
                                              rocblas_int       ldc,
                                              rocblas_stride    strideC,
                                              rocblas_int       batch_count)
{
    if(uplo != rocblas_fill_lower && uplo != rocblas_fill_upper)
        return rocblas_status_invalid_value;

    if(trans != rocblas_operation_none && trans != rocblas_operation_transpose)
        return rocblas_status_invalid_value;

    if(n < 0 || k < 0 || batch_count < 0 || ldc < n
       || (trans == rocblas_operation_none && (lda < n || ldb < n))
       || (trans != rocblas_operation_none && (lda < k || ldb < k)))
        return rocblas_status_invalid_size;

    if(!n || !batch_count)
        return rocblas_status_success;

    if((k > 0 && (!AP || !BP || !alpha)) || !CP || !beta)
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <typename TScal, typename TConstPtr, typename UScal, typename TPtr>
inline rocblas_status rocblas_her2k_arg_check(rocblas_handle    handle,
                                              rocblas_fill      uplo,
                                              rocblas_operation trans,
                                              rocblas_int       n,
                                              rocblas_int       k,
                                              TScal             alpha,
                                              TConstPtr         AP,
                                              rocblas_stride    offsetA,
                                              rocblas_int       lda,
                                              rocblas_stride    strideA,
                                              TConstPtr         BP,
                                              rocblas_stride    offsetB,
                                              rocblas_int       ldb,
                                              rocblas_stride    strideB,
                                              UScal             beta,
                                              TPtr              CP,
                                              rocblas_stride    offsetC,
                                              rocblas_int       ldc,
                                              rocblas_stride    strideC,
                                              rocblas_int       batch_count)
{
    if(uplo != rocblas_fill_lower && uplo != rocblas_fill_upper)
        return rocblas_status_invalid_value;

    if(trans != rocblas_operation_none && trans != rocblas_operation_conjugate_transpose)
        return rocblas_status_invalid_value;

    if(n < 0 || k < 0 || batch_count < 0 || ldc < n
       || (trans == rocblas_operation_none && (lda < n || ldb < n))
       || (trans != rocblas_operation_none && (lda < k || ldb < k)))
        return rocblas_status_invalid_size;

    if(!n || !batch_count)
        return rocblas_status_success;

    if((k > 0 && (!AP || !BP || !alpha)) || !CP || !beta)
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

/**
  *  TScal     is always: const T* (either host or device)
  *  TConstPtr is either: const T* OR const T* const*
  *  TPtr      is either:       T* OR       T* const*
  */
template <bool BATCHED, bool TWOK, typename TScal, typename TConstPtr, typename TPtr>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_syr2k_template(rocblas_handle    handle,
                                    rocblas_fill      uplo,
                                    rocblas_operation trans,
                                    rocblas_int       n,
                                    rocblas_int       k,
                                    TScal             alpha,
                                    TConstPtr         AP,
                                    rocblas_stride    offsetA,
                                    rocblas_int       lda,
                                    rocblas_stride    strideA,
                                    TConstPtr         BP,
                                    rocblas_stride    offsetB,
                                    rocblas_int       ldb,
                                    rocblas_stride    strideB,
                                    TScal             beta,
                                    TPtr              CP,
                                    rocblas_stride    offsetC,
                                    rocblas_int       ldc,
                                    rocblas_stride    strideC,
                                    rocblas_int       batch_count);

/**
  *  TScal     is always: const T* (either host or device)
  *  TConstPtr is either: const T* OR const T* const*
  *  TPtr      is either:       T* OR       T* const*
  */
template <bool BATCHED,
          bool TWOK,
          typename TScal,
          typename TConstPtr,
          typename UScal,
          typename TPtr>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_her2k_template(rocblas_handle    handle,
                                    rocblas_fill      uplo,
                                    rocblas_operation trans,
                                    rocblas_int       n,
                                    rocblas_int       k,
                                    TScal             alpha,
                                    TConstPtr         AP,
                                    rocblas_stride    offsetA,
                                    rocblas_int       lda,
                                    rocblas_stride    strideA,
                                    TConstPtr         BP,
                                    rocblas_stride    offsetB,
                                    rocblas_int       ldb,
                                    rocblas_stride    strideB,
                                    UScal             beta,
                                    TPtr              CP,
                                    rocblas_stride    offsetC,
                                    rocblas_int       ldc,
                                    rocblas_stride    strideC,
                                    rocblas_int       batch_count);
