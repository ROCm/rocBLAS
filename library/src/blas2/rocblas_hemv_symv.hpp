/* ************************************************************************
 * Copyright (C) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include "check_numerics_vector.hpp"
#include "handle.hpp"

template <typename To>
ROCBLAS_INTERNAL_EXPORT_NOINLINE size_t
    rocblas_internal_hemv_symv_kernel_workspace_size(rocblas_int n, rocblas_int batch_count = 1);

template <typename T, typename U, typename V, typename TPtr>
inline rocblas_status rocblas_hemv_symv_arg_check(rocblas_handle handle,
                                                  rocblas_fill   uplo,
                                                  rocblas_int    n,
                                                  const V*       alpha,
                                                  rocblas_stride stride_alpha,
                                                  const U*       A,
                                                  rocblas_stride offseta,
                                                  rocblas_int    lda,
                                                  rocblas_stride strideA,
                                                  const U*       x,
                                                  rocblas_stride offsetx,
                                                  rocblas_int    incx,
                                                  rocblas_stride stridex,
                                                  const V*       beta,
                                                  rocblas_stride stride_beta,
                                                  const TPtr*    y,
                                                  rocblas_stride offsety,
                                                  rocblas_int    incy,
                                                  rocblas_stride stridey,
                                                  rocblas_int    batch_count)
{
    if(uplo != rocblas_fill_lower && uplo != rocblas_fill_upper)
        return rocblas_status_invalid_value;

    if(n < 0 || lda < n || lda < 1 || !incx || !incy || batch_count < 0)
        return rocblas_status_invalid_size;

    if(!n || !batch_count)
        return rocblas_status_success;

    if(!beta || !alpha)
        return rocblas_status_invalid_pointer;

    if(handle->pointer_mode == rocblas_pointer_mode_host)
    {
        // only supports stride_alpha and stride_beta for device memory alpha/beta
        if(stride_alpha || stride_beta)
            return rocblas_status_not_implemented;

        if(*alpha == 0 && *beta == 1)
            return rocblas_status_success;

        if(!y || (*alpha != 0 && (!A || !x)))
            return rocblas_status_invalid_pointer;
    }

    return rocblas_status_continue;
}

/**
  *  V is either: const T* OR const T* const*
  *  W is either:       T* OR       T* const*
  *  Note stride_alpha and stride_beta are only used AND only tested by rocSOLVER
  *  These strided scalar fetches are only supported for device_ptr mode
  */
template <bool IS_HEMV, typename U, typename V, typename TPtr, typename W>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_hemv_symv_template(rocblas_handle handle,
                                        rocblas_fill   uplo,
                                        rocblas_int    n,
                                        const U*       alpha,
                                        rocblas_stride stride_alpha,
                                        V              A,
                                        rocblas_stride offseta,
                                        rocblas_int    lda,
                                        rocblas_stride strideA,
                                        V              x,
                                        rocblas_stride offsetx,
                                        rocblas_int    incx,
                                        rocblas_stride stridex,
                                        const U*       beta,
                                        rocblas_stride stride_beta,
                                        TPtr           y,
                                        rocblas_stride offsety,
                                        rocblas_int    incy,
                                        rocblas_stride stridey,
                                        rocblas_int    batch_count,
                                        W              workspace);

template <typename T, typename U, typename V, typename TPtr, typename W>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_symv_template(rocblas_handle handle,
                                   rocblas_fill   uplo,
                                   rocblas_int    n,
                                   const V*       alpha,
                                   rocblas_stride stride_alpha,
                                   const U*       A,
                                   rocblas_stride offseta,
                                   rocblas_int    lda,
                                   rocblas_stride strideA,
                                   const U*       x,
                                   rocblas_stride offsetx,
                                   rocblas_int    incx,
                                   rocblas_stride stridex,
                                   const V*       beta,
                                   rocblas_stride stride_beta,
                                   TPtr*          y,
                                   rocblas_stride offsety,
                                   rocblas_int    incy,
                                   rocblas_stride stridey,
                                   rocblas_int    batch_count,
                                   W              workspace);

//TODO :-Add rocblas_check_numerics_he_matrix_template for checking Matrix `A` which is a Hermitian Matrix
template <typename T, typename U>
rocblas_status rocblas_hemv_check_numerics(const char*    function_name,
                                           rocblas_handle handle,
                                           rocblas_fill   uplo,
                                           rocblas_int    n,
                                           T              A,
                                           rocblas_stride offset_a,
                                           rocblas_int    lda,
                                           rocblas_stride stride_a,
                                           T              x,
                                           rocblas_stride offset_x,
                                           rocblas_int    inc_x,
                                           rocblas_stride stride_x,
                                           U              y,
                                           rocblas_stride offset_y,
                                           rocblas_int    inc_y,
                                           rocblas_stride stride_y,
                                           rocblas_int    batch_count,
                                           const int      check_numerics,
                                           bool           is_input);
template <typename T, typename U>
rocblas_status rocblas_symv_check_numerics(const char*    function_name,
                                           rocblas_handle handle,
                                           rocblas_fill   uplo,
                                           rocblas_int    n,
                                           T              A,
                                           rocblas_stride offset_a,
                                           rocblas_int    lda,
                                           rocblas_stride stride_a,
                                           T              x,
                                           rocblas_stride offset_x,
                                           rocblas_int    inc_x,
                                           rocblas_stride stride_x,
                                           U              y,
                                           rocblas_stride offset_y,
                                           rocblas_int    inc_y,
                                           rocblas_stride stride_y,
                                           rocblas_int    batch_count,
                                           const int      check_numerics,
                                           bool           is_input);
