/* ************************************************************************
 * Copyright (C) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
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
ROCBLAS_KERNEL(DIM_XYT* DIM_XYT)
syr2k_her2k_kernel(bool              upper,
                   rocblas_operation trans,
                   rocblas_int       n,
                   rocblas_int       k,
                   TScal             alpha_host_device,
                   TConstPtr         AP_array,
                   rocblas_stride    shift_a,
                   rocblas_int       lda,
                   rocblas_stride    stride_a,
                   TConstPtr         BP_array,
                   rocblas_stride    shift_b,
                   rocblas_int       ldb,
                   rocblas_stride    stride_b,
                   TPtr              CP_array,
                   rocblas_stride    shift_c,
                   rocblas_int       ldc,
                   rocblas_stride    stride_c);

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

    if(std::is_same<TScal, const rocblas_float*>{} || std::is_same<TScal, const rocblas_double*>{})
    {
        // ssyr2k and dsyr2k all forms
        if(trans != rocblas_operation_none && trans != rocblas_operation_transpose
           && trans != rocblas_operation_conjugate_transpose)
            return rocblas_status_invalid_value;
    }
    else
    {
        if(trans != rocblas_operation_none && trans != rocblas_operation_transpose)
            return rocblas_status_invalid_value;
    }

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
