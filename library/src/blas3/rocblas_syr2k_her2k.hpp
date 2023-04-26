/* ************************************************************************
 * Copyright (C) 2020-2023 Advanced Micro Devices, Inc. All rights reserved.
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
#include "herk_syrk_device.hpp"

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

    if(std::is_same_v<TScal, const rocblas_float*> || std::is_same_v<TScal, const rocblas_double*>)
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

    if((k > 0 && !alpha) || !beta)
        return rocblas_status_invalid_pointer;

    if(handle->pointer_mode == rocblas_pointer_mode_host)
    {
        bool calcAB = k > 0 && *alpha != 0;

        if(!calcAB && *beta == 1)
            return rocblas_status_success;

        if((calcAB && (!AP || !BP)) || ((calcAB || *beta != 1) && !CP))
            return rocblas_status_invalid_pointer;
    }

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

    if((k > 0 && !alpha) || !beta)
        return rocblas_status_invalid_pointer;

    if(handle->pointer_mode == rocblas_pointer_mode_host)
    {
        bool calcAB = k > 0 && *alpha != 0;

        if(!calcAB && *beta == 1)
            return rocblas_status_success; // avoid slow kernel launches for no op

        if((calcAB && (!AP || !BP)) || ((calcAB || *beta != 1) && !CP))
            return rocblas_status_invalid_pointer;
    }

    return rocblas_status_continue;
}

template <rocblas_int MIN_NB,
          bool        BATCHED,
          bool        TWOK,
          bool        HERK,
          typename T,
          typename TScala,
          typename TScalb,
          typename TConstPtr,
          typename TPtr>
rocblas_status rocblas_internal_syr2k_her2k_template(rocblas_handle    handle,
                                                     rocblas_fill      uplo,
                                                     rocblas_operation trans,
                                                     rocblas_int       n,
                                                     rocblas_int       k,
                                                     const TScala*     alpha,
                                                     TConstPtr         dA_in,
                                                     rocblas_stride    offset_a,
                                                     rocblas_int       lda,
                                                     rocblas_stride    stride_a,
                                                     TConstPtr         dB_in,
                                                     rocblas_stride    offset_b,
                                                     rocblas_int       ldb,
                                                     rocblas_stride    stride_b,
                                                     const TScalb*     beta,
                                                     TPtr              dC_in,
                                                     rocblas_stride    offset_c,
                                                     rocblas_int       ldc,
                                                     rocblas_stride    stride_c,
                                                     rocblas_int       batch_count);

template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_syr2k_template(rocblas_handle    handle,
                                    rocblas_fill      uplo,
                                    rocblas_operation trans,
                                    rocblas_int       n,
                                    rocblas_int       k,
                                    const T*          alpha,
                                    const T*          dA_in,
                                    rocblas_stride    offset_a,
                                    rocblas_int       lda,
                                    rocblas_stride    stride_a,
                                    const T*          dB_in,
                                    rocblas_stride    offset_b,
                                    rocblas_int       ldb,
                                    rocblas_stride    stride_b,
                                    const T*          beta,
                                    T*                dC_in,
                                    rocblas_stride    offset_c,
                                    rocblas_int       ldc,
                                    rocblas_stride    stride_c,
                                    rocblas_int       batch_count);

template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_syr2k_batched_template(rocblas_handle    handle,
                                            rocblas_fill      uplo,
                                            rocblas_operation trans,
                                            rocblas_int       n,
                                            rocblas_int       k,
                                            const T*          alpha,
                                            const T* const*   dA_in,
                                            rocblas_stride    offset_a,
                                            rocblas_int       lda,
                                            rocblas_stride    stride_a,
                                            const T* const*   dB_in,
                                            rocblas_stride    offset_b,
                                            rocblas_int       ldb,
                                            rocblas_stride    stride_b,
                                            const T*          beta,
                                            T* const*         dC_in,
                                            rocblas_stride    offset_c,
                                            rocblas_int       ldc,
                                            rocblas_stride    stride_c,
                                            rocblas_int       batch_count);

template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_her2k_template(rocblas_handle    handle,
                                    rocblas_fill      uplo,
                                    rocblas_operation trans,
                                    rocblas_int       n,
                                    rocblas_int       k,
                                    const T*          alpha,
                                    const T*          dA_in,
                                    rocblas_stride    offset_a,
                                    rocblas_int       lda,
                                    rocblas_stride    stride_a,
                                    const T*          dB_in,
                                    rocblas_stride    offset_b,
                                    rocblas_int       ldb,
                                    rocblas_stride    stride_b,
                                    const real_t<T>*  beta,
                                    T*                dC_in,
                                    rocblas_stride    offset_c,
                                    rocblas_int       ldc,
                                    rocblas_stride    stride_c,
                                    rocblas_int       batch_count);

template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_her2k_batched_template(rocblas_handle    handle,
                                            rocblas_fill      uplo,
                                            rocblas_operation trans,
                                            rocblas_int       n,
                                            rocblas_int       k,
                                            const T*          alpha,
                                            const T* const*   dA_in,
                                            rocblas_stride    offset_a,
                                            rocblas_int       lda,
                                            rocblas_stride    stride_a,
                                            const T* const*   dB_in,
                                            rocblas_stride    offset_b,
                                            rocblas_int       ldb,
                                            rocblas_stride    stride_b,
                                            const real_t<T>*  beta,
                                            T* const*         dC_in,
                                            rocblas_stride    offset_c,
                                            rocblas_int       ldc,
                                            rocblas_stride    stride_c,
                                            rocblas_int       batch_count);

template <bool HERM, typename TConstPtr, typename TPtr>
rocblas_status rocblas_her2k_syr2k_check_numerics(const char*       function_name,
                                                  rocblas_handle    handle,
                                                  rocblas_fill      uplo,
                                                  rocblas_operation trans,
                                                  rocblas_int       n,
                                                  rocblas_int       k,
                                                  TConstPtr         A,
                                                  rocblas_int       lda,
                                                  rocblas_stride    strideA,
                                                  TConstPtr         B,
                                                  rocblas_int       ldb,
                                                  rocblas_stride    strideB,
                                                  TPtr              C,
                                                  rocblas_int       ldc,
                                                  rocblas_stride    strideC,
                                                  rocblas_int       batch_count,
                                                  const int         check_numerics,
                                                  bool              is_input);

template <bool TWOK, bool HERM, rocblas_int DIM_XYT, typename T, typename TConstPtr, typename TPtr>
void syr2k_her2k_dispatch(rocblas_fill      uplo,
                          rocblas_operation trans,
                          rocblas_int       n,
                          rocblas_int       k,
                          const T           alpha,
                          TConstPtr*        dA,
                          rocblas_int       lda,
                          rocblas_stride    stride_a,
                          TConstPtr*        dB,
                          rocblas_int       ldb,
                          rocblas_stride    stride_b,
                          TPtr*             dC,
                          rocblas_int       ldc,
                          rocblas_stride    stride_c,
                          rocblas_int       batch_count,
                          hipStream_t       stream);
