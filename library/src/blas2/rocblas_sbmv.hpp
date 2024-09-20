/* ************************************************************************
 * Copyright (C) 2019-2024 Advanced Micro Devices, Inc. All rights reserved.
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

template <typename API_INT, typename T, typename U, typename V, typename W>
inline rocblas_status rocblas_sbmv_arg_check(rocblas_handle handle,
                                             rocblas_fill   uplo,
                                             API_INT        n,
                                             API_INT        k,
                                             const V*       alpha,
                                             rocblas_stride stride_alpha,
                                             const U*       A,
                                             rocblas_stride offseta,
                                             API_INT        lda,
                                             rocblas_stride strideA,
                                             const U*       x,
                                             rocblas_stride offsetx,
                                             API_INT        incx,
                                             rocblas_stride stridex,
                                             const V*       beta,
                                             rocblas_stride stride_beta,
                                             const W*       y,
                                             rocblas_stride offsety,
                                             API_INT        incy,
                                             rocblas_stride stridey,
                                             API_INT        batch_count)
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

    if(n < 0 || k < 0 || lda < k + 1 || lda < 1 || !incx || !incy || batch_count < 0)
        return rocblas_status_invalid_size;

    // quick return before pointer checks
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

        // pointers are validated if they need to be dereferenced
        if(!y || (*alpha != 0 && (!A || !x)))
            return rocblas_status_invalid_pointer;
    }

    return rocblas_status_continue;
}

template <typename T, typename TScal, typename TConstPtr, typename TPtr>
rocblas_status rocblas_internal_sbmv_launcher(rocblas_handle handle,
                                              rocblas_fill   uplo,
                                              rocblas_int    n,
                                              rocblas_int    k,
                                              TScal          alpha,
                                              rocblas_stride stride_alpha,
                                              TConstPtr      A,
                                              rocblas_stride offseta,
                                              int64_t        lda,
                                              rocblas_stride strideA,
                                              TConstPtr      x,
                                              rocblas_stride offsetx,
                                              int64_t        incx,
                                              rocblas_stride stridex,
                                              TScal          beta,
                                              rocblas_stride stride_beta,
                                              TPtr           y,
                                              rocblas_stride offsety,
                                              int64_t        incy,
                                              rocblas_stride stridey,
                                              rocblas_int    batch_count);

//TODO :-Add rocblas_check_numerics_sb_matrix_template for checking Matrix `A` which is a Symmetric Band Matrix
template <typename T, typename U>
rocblas_status rocblas_sbmv_check_numerics(const char*    function_name,
                                           rocblas_handle handle,
                                           int64_t        n,
                                           T              A,
                                           rocblas_stride offset_a,
                                           int64_t        lda,
                                           rocblas_stride stride_a,
                                           T              x,
                                           rocblas_stride offset_x,
                                           int64_t        inc_x,
                                           rocblas_stride stride_x,
                                           U              y,
                                           rocblas_stride offset_y,
                                           int64_t        inc_y,
                                           rocblas_stride stride_y,
                                           int64_t        batch_count,
                                           const int      check_numerics,
                                           bool           is_input);
