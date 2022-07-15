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

#include "rocblas.h"

#include "handle.hpp"

/**
  *
  * rocblas_check_numerics_vector_kernel(n, xa, offset_x, inc_x, stride_x, abnormal)
  *
  * Info about rocblas_check_numerics_vector_kernel function:
  *
  *    It is the kernel function which checks a vector for numerical abnormalities such as NaN/zero/Inf/denormal values and updates the rocblas_check_numerics_t structure.
  *
  * Parameters   : n            : Total number of elements in the vector
  *                xa           : Pointer to the vector which is under consideration for numerical abnormalities
  *                offset_x     : Offset of vector 'xa'
  *                inc_x        : Stride between consecutive values of vector 'xa'
  *                stride_x     : Specifies the pointer increment between one vector 'x_i' and the next one (xa_i+1) (where (xa_i) is the i-th instance of the batch)
  *                abnormal     : Device pointer to the rocblas_check_numerics_t structure
  *
  * Return Value : Nothing --
  *
**/

template <int DIM_X, typename T>
ROCBLAS_KERNEL(DIM_X)
rocblas_check_numerics_vector_kernel(rocblas_int               n,
                                     T                         xa,
                                     rocblas_stride            offset_x,
                                     rocblas_int               inc_x,
                                     rocblas_stride            stride_x,
                                     rocblas_check_numerics_t* abnormal)
{
    auto*     x   = load_ptr_batch(xa, hipBlockIdx_y, offset_x, stride_x);
    ptrdiff_t tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    //Check every element of the x vector for a NaN/zero/Inf/denormal value
    if(tid < n)
    {
        auto value = x[tid * inc_x];
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

rocblas_status rocblas_check_numerics_abnormal_struct(const char*               function_name,
                                                      const int                 check_numerics,
                                                      bool                      is_input,
                                                      rocblas_check_numerics_t* h_abnormal);

template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_check_numerics_vector_template(const char*    function_name,
                                                    rocblas_handle handle,
                                                    rocblas_int    n,
                                                    T              x,
                                                    rocblas_stride offset_x,
                                                    rocblas_int    inc_x,
                                                    rocblas_stride stride_x,
                                                    rocblas_int    batch_count,
                                                    const int      check_numerics,
                                                    bool           is_input);
