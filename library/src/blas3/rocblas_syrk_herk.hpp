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

template <typename TScal, typename TConstPtr, typename TPtr>
inline rocblas_status rocblas_syrk_arg_check(rocblas_handle    handle,
                                             rocblas_fill      uplo,
                                             rocblas_operation transA,
                                             rocblas_int       n,
                                             rocblas_int       k,
                                             const TScal*      alpha,
                                             TConstPtr         AP,
                                             rocblas_stride    offsetA,
                                             rocblas_int       lda,
                                             rocblas_stride    strideA,
                                             const TScal*      beta,
                                             TPtr              CP,
                                             rocblas_stride    offsetC,
                                             rocblas_int       ldc,
                                             rocblas_stride    strideC,
                                             rocblas_int       batch_count)
{
    if(uplo != rocblas_fill_lower && uplo != rocblas_fill_upper)
        return rocblas_status_invalid_value;

    if(is_complex<TScal>)
    {
        if(transA != rocblas_operation_none && transA != rocblas_operation_transpose)
            return rocblas_status_invalid_value;
    }
    else
    {
        if(transA != rocblas_operation_none && transA != rocblas_operation_transpose
           && transA != rocblas_operation_conjugate_transpose)
            return rocblas_status_invalid_value;
    }

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
                                             rocblas_stride    offsetA,
                                             rocblas_int       lda,
                                             rocblas_stride    strideA,
                                             TScal             beta,
                                             TPtr              CP,
                                             rocblas_stride    offsetC,
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
                                   rocblas_stride    offsetA,
                                   rocblas_int       lda,
                                   rocblas_stride    strideA,
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
template <typename TScal, typename TConstPtr, typename TPtr>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_herk_template(rocblas_handle    handle,
                                   rocblas_fill      uplo,
                                   rocblas_operation transA,
                                   rocblas_int       n,
                                   rocblas_int       k,
                                   TScal             alpha,
                                   TConstPtr         AP,
                                   rocblas_stride    offsetA,
                                   rocblas_int       lda,
                                   rocblas_stride    strideA,
                                   TScal             beta,
                                   TPtr              CP,
                                   rocblas_stride    offsetC,
                                   rocblas_int       ldc,
                                   rocblas_stride    strideC,
                                   rocblas_int       batch_count);
