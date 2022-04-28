/* ************************************************************************
 * Copyright (C) 2016-2022 Advanced Micro Devices, Inc. All rights reserved.
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
#include "rocblas.h"

/**
  *
  * rocblas_check_numerics_ge_matrix_kernel(m, n, Aa, offset_a, lda, stride_a, abnormal)
  *
  * Info about rocblas_check_numerics_ge_matrix_kernel function:
  *
  *    It is the kernel function which checks a matrix for numerical abnormalities such as NaN/zero/Inf/denormal values and updates the rocblas_check_numerics_t structure.
  *    ge in rocblas_check_numerics_ge_matrix_kernel refers to general.
  *
  * Parameters   : m            : number of rows of matrix 'A'
  *                n            : number of columns of matrix 'A'
  *                Aa           : Pointer to the matrix which is under consideration for numerical abnormalities
  *                offset_a     : Offset of matrix 'Aa'
  *                lda          : specifies the leading dimension of matrix 'Aa'
  *                stride_a     : Specifies the pointer increment between one matrix 'A_i' and the next one (Aa_i+1) (where (Aa_i) is the i-th instance of the batch)
  *                abnormal     : Device pointer to the rocblas_check_numerics_t structure
  *
  * Return Value : Nothing --
  *
**/

template <typename T>
ROCBLAS_KERNEL_NO_BOUNDS rocblas_check_numerics_ge_matrix_kernel(rocblas_int               m,
                                                                 rocblas_int               n,
                                                                 T                         Aa,
                                                                 rocblas_stride            offset_a,
                                                                 rocblas_int               lda,
                                                                 rocblas_stride            stride_a,
                                                                 rocblas_check_numerics_t* abnormal)
{
    rocblas_int tx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    rocblas_int ty = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    //Check every element of the A matrix for a NaN/zero/Inf/denormal value
    if(tx < m && ty < n)
    {
        auto* A = load_ptr_batch(Aa, hipBlockIdx_z, offset_a, stride_a);

        ptrdiff_t tid   = tx + ptrdiff_t(lda) * ty;
        auto      value = A[tid];
        if(!abnormal->has_zero && rocblas_iszero(value))
            abnormal->has_zero = true;
        if(!abnormal->has_NaN && rocblas_isnan(value))
            abnormal->has_NaN = true;
        if(!abnormal->has_Inf && rocblas_isinf(value))
            abnormal->has_Inf = true;
        if(!abnormal->has_denorm && rocblas_isdenorm(value))
            abnormal->has_denorm = true;
    }
}
template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_check_numerics_ge_matrix_template(const char*       function_name,
                                                       rocblas_handle    handle,
                                                       rocblas_operation trans_a,
                                                       rocblas_int       m,
                                                       rocblas_int       n,
                                                       T                 A,
                                                       rocblas_stride    offset_a,
                                                       rocblas_int       lda,
                                                       rocblas_stride    stride_a,
                                                       rocblas_int       batch_count,
                                                       const int         check_numerics,
                                                       bool              is_input);
