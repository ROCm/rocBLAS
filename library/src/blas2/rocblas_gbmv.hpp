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

#include "../blas1/rocblas_copy.hpp"
#include "check_numerics_matrix.hpp"
#include "check_numerics_vector.hpp"

template <typename T, typename U, typename V>
inline rocblas_status rocblas_gbmv_arg_check(rocblas_handle    handle,
                                             rocblas_operation transA,
                                             rocblas_int       m,
                                             rocblas_int       n,
                                             rocblas_int       kl,
                                             rocblas_int       ku,
                                             const T*          alpha,
                                             U                 A,
                                             rocblas_stride    offseta,
                                             rocblas_int       lda,
                                             rocblas_stride    strideA,
                                             U                 x,
                                             rocblas_stride    offsetx,
                                             rocblas_int       incx,
                                             rocblas_stride    stridex,
                                             const T*          beta,
                                             V                 y,
                                             rocblas_stride    offsety,
                                             rocblas_int       incy,
                                             rocblas_stride    stridey,
                                             rocblas_int       batch_count)
{
    if(transA != rocblas_operation_none && transA != rocblas_operation_transpose
       && transA != rocblas_operation_conjugate_transpose)
        return rocblas_status_invalid_value;

    if(m < 0 || n < 0 || lda < ku + kl + 1 || !incx || !incy || kl < 0 || ku < 0 || batch_count < 0)
        return rocblas_status_invalid_size;

    if(!m || !n || !batch_count)
        return rocblas_status_success;

    if(!alpha || !beta)
        return rocblas_status_invalid_pointer;

    if(handle->pointer_mode == rocblas_pointer_mode_host)
    {
        if(*alpha == 0 && *beta == 1)
            return rocblas_status_success;

        if(!y || (*alpha != 0 && (!A || !x)))
            return rocblas_status_invalid_pointer;
    }

    return rocblas_status_continue;
}

/**
  *  Here, U is either a `const T* const*` or a `const T*`
  *  V is either a `T*` or a `T* const*`
  */
template <typename T, typename U, typename V>
rocblas_status rocblas_gbmv_template(rocblas_handle    handle,
                                     rocblas_operation transA,
                                     rocblas_int       m,
                                     rocblas_int       n,
                                     rocblas_int       kl,
                                     rocblas_int       ku,
                                     const T*          alpha,
                                     U                 A,
                                     rocblas_stride    offseta,
                                     rocblas_int       lda,
                                     rocblas_stride    strideA,
                                     U                 x,
                                     rocblas_stride    offsetx,
                                     rocblas_int       incx,
                                     rocblas_stride    stridex,
                                     const T*          beta,
                                     V                 y,
                                     rocblas_stride    offsety,
                                     rocblas_int       incy,
                                     rocblas_stride    stridey,
                                     rocblas_int       batch_count);

//TODO :-Add rocblas_check_numerics_gb_matrix_template for checking Matrix `A` which is a General Band matrix
template <typename T, typename U>
rocblas_status rocblas_gbmv_check_numerics(const char*       function_name,
                                           rocblas_handle    handle,
                                           rocblas_operation trans_a,
                                           rocblas_int       m,
                                           rocblas_int       n,
                                           T                 A,
                                           rocblas_stride    offset_a,
                                           rocblas_int       lda,
                                           rocblas_stride    stride_a,
                                           T                 x,
                                           rocblas_stride    offset_x,
                                           rocblas_int       inc_x,
                                           rocblas_stride    stride_x,
                                           U                 y,
                                           rocblas_stride    offset_y,
                                           rocblas_int       inc_y,
                                           rocblas_stride    stride_y,
                                           rocblas_int       batch_count,
                                           const int         check_numerics,
                                           bool              is_input);
