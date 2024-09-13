/* ************************************************************************
 * Copyright (C) 2020-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "check_numerics_matrix.hpp"
#include "handle.hpp"

template <typename API_INT, typename TScal, typename TConstPtr, typename TPtr>
inline rocblas_status rocblas_syrk_arg_check(rocblas_handle    handle,
                                             rocblas_fill      uplo,
                                             rocblas_operation transA,
                                             API_INT           n,
                                             API_INT           k,
                                             const TScal*      alpha,
                                             TConstPtr         AP,
                                             rocblas_stride    offsetA,
                                             API_INT           lda,
                                             rocblas_stride    strideA,
                                             const TScal*      beta,
                                             TPtr              CP,
                                             rocblas_stride    offsetC,
                                             API_INT           ldc,
                                             rocblas_stride    strideC,
                                             API_INT           batch_count)
{
    if constexpr(std::is_same_v<API_INT, int>)
    {
        if(batch_count > c_YZ_grid_launch_limit && handle->isYZGridDim16bit())
        {
            return rocblas_status_invalid_size;
        }
    }

    if(uplo != rocblas_fill_lower && uplo != rocblas_fill_upper)
        return rocblas_status_invalid_value;

    if(rocblas_is_complex<TScal>)
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

    if((k > 0 && !alpha) || !beta)
        return rocblas_status_invalid_pointer;

    if(handle->pointer_mode == rocblas_pointer_mode_host)
    {
        bool calcA = k > 0 && *alpha != 0;

        if(!calcA && *beta == 1)
            return rocblas_status_success; // avoid slow kernel launches for no op

        if((calcA && !AP) || ((calcA || *beta != 1) && !CP))
            return rocblas_status_invalid_pointer;
    }

    return rocblas_status_continue;
}

template <typename API_INT, typename TScal, typename TConstPtr, typename TPtr>
inline rocblas_status rocblas_herk_arg_check(rocblas_handle    handle,
                                             rocblas_fill      uplo,
                                             rocblas_operation transA,
                                             API_INT           n,
                                             API_INT           k,
                                             TScal             alpha,
                                             TConstPtr         AP,
                                             rocblas_stride    offsetA,
                                             API_INT           lda,
                                             rocblas_stride    strideA,
                                             TScal             beta,
                                             TPtr              CP,
                                             rocblas_stride    offsetC,
                                             API_INT           ldc,
                                             rocblas_stride    strideC,
                                             API_INT           batch_count)
{
    if constexpr(std::is_same_v<API_INT, int>)
    {
        if(batch_count > c_YZ_grid_launch_limit && handle->isYZGridDim16bit())
        {
            return rocblas_status_invalid_size;
        }
    }

    if(uplo != rocblas_fill_lower && uplo != rocblas_fill_upper)
        return rocblas_status_invalid_value;
    if(transA != rocblas_operation_none && transA != rocblas_operation_conjugate_transpose)
        return rocblas_status_invalid_value;

    if(n < 0 || k < 0 || batch_count < 0 || ldc < n || (transA == rocblas_operation_none && lda < n)
       || (transA != rocblas_operation_none && lda < k))
        return rocblas_status_invalid_size;

    if(!n || !batch_count)
        return rocblas_status_success;

    if((k > 0 && !alpha) || !beta)
        return rocblas_status_invalid_pointer;

    if(handle->pointer_mode == rocblas_pointer_mode_host)
    {
        bool calcA = k > 0 && *alpha != 0;

        if(!calcA && *beta == 1)
            return rocblas_status_success; // avoid slow kernel launches for no op

        if((calcA && !AP) || ((calcA || *beta != 1) && !CP))
            return rocblas_status_invalid_pointer;
    }

    return rocblas_status_continue;
}

template <bool HERM, typename TConstPtr, typename TPtr>
rocblas_status rocblas_herk_syrk_check_numerics(const char*       function_name,
                                                rocblas_handle    handle,
                                                rocblas_fill      uplo,
                                                rocblas_operation trans,
                                                int64_t           n_64,
                                                int64_t           k_64,
                                                TConstPtr         A,
                                                int64_t           lda_64,
                                                rocblas_stride    strideA,
                                                TPtr              C,
                                                int64_t           ldc_64,
                                                rocblas_stride    strideC,
                                                int64_t           batch_count_64,
                                                const int         check_numerics,
                                                bool              is_input);

/*
 * internal rocBLAS template function, also used by rocSOLVER.
 * Used for calls to rocblas_xsyrk() and rocblas_xsyrk_strided_batched()
 */
template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_syrk_template(rocblas_handle    handle,
                                   rocblas_fill      uplo,
                                   rocblas_operation transA,
                                   rocblas_int       n,
                                   rocblas_int       k,
                                   const T*          alpha,
                                   const T*          A,
                                   rocblas_stride    offsetA,
                                   rocblas_int       lda,
                                   rocblas_stride    strideA,
                                   const T*          beta,
                                   T*                C,
                                   rocblas_stride    offsetC,
                                   rocblas_int       ldc,
                                   rocblas_stride    strideC,
                                   rocblas_int       batch_count);

/*
 * internal rocBLAS template function, also used by rocSOLVER.
 * Used for calls to rocblas_xsyrk_batched()
 */
template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_syrk_batched_template(rocblas_handle    handle,
                                           rocblas_fill      uplo,
                                           rocblas_operation transA,
                                           rocblas_int       n,
                                           rocblas_int       k,
                                           const T*          alpha,
                                           const T* const*   A,
                                           rocblas_stride    offsetA,
                                           rocblas_int       lda,
                                           rocblas_stride    strideA,
                                           const T*          beta,
                                           T* const*         C,
                                           rocblas_stride    offsetC,
                                           rocblas_int       ldc,
                                           rocblas_stride    strideC,
                                           rocblas_int       batch_count);

/*
 * internal rocBLAS template function, also used by rocSOLVER.
 * Used for calls to rocblas_xherk() and rocblas_xherk_strided_batched()
 */
template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_herk_template(rocblas_handle    handle,
                                   rocblas_fill      uplo,
                                   rocblas_operation transA,
                                   rocblas_int       n,
                                   rocblas_int       k,
                                   const real_t<T>*  alpha,
                                   const T*          A,
                                   rocblas_stride    offsetA,
                                   rocblas_int       lda,
                                   rocblas_stride    strideA,
                                   const real_t<T>*  beta,
                                   T*                C,
                                   rocblas_stride    offsetC,
                                   rocblas_int       ldc,
                                   rocblas_stride    strideC,
                                   rocblas_int       batch_count);

/*
 * internal rocBLAS template function, also used by rocSOLVER.
 * Used for calls to rocblas_xherk_batched()
 */
template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_herk_batched_template(rocblas_handle    handle,
                                           rocblas_fill      uplo,
                                           rocblas_operation transA,
                                           rocblas_int       n,
                                           rocblas_int       k,
                                           const real_t<T>*  alpha,
                                           const T* const*   A,
                                           rocblas_stride    offsetA,
                                           rocblas_int       lda,
                                           rocblas_stride    strideA,
                                           const real_t<T>*  beta,
                                           T* const*         C,
                                           rocblas_stride    offsetC,
                                           rocblas_int       ldc,
                                           rocblas_stride    strideC,
                                           rocblas_int       batch_count);
