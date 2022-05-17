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
#include "check_numerics_matrix.hpp"
#include "check_numerics_vector.hpp"
#include "handle.hpp"

/**
 * TConstPtr is either: const T* OR const T* const*
 * TPtr      is either:       T* OR       T* const*
 * Where T is the base type (float, double, rocblas_complex, or rocblas_double_complex)
 */

template <typename TConstPtr, typename TPtr>
rocblas_status rocblas_dgmm_template(rocblas_handle handle,
                                     rocblas_side   side,
                                     rocblas_int    m,
                                     rocblas_int    n,
                                     TConstPtr      A,
                                     rocblas_stride offset_a,
                                     rocblas_int    lda,
                                     rocblas_stride stride_a,
                                     TConstPtr      X,
                                     rocblas_stride offset_x,
                                     rocblas_int    incx,
                                     rocblas_stride stride_x,
                                     TPtr           C,
                                     rocblas_stride offset_c,
                                     rocblas_int    ldc,
                                     rocblas_stride stride_c,
                                     rocblas_int    batch_count);

template <typename TConstPtr, typename TPtr>
rocblas_status rocblas_dgmm_check_numerics(const char*    function_name,
                                           rocblas_handle handle,
                                           rocblas_side   side,
                                           rocblas_int    m,
                                           rocblas_int    n,
                                           TConstPtr      A,
                                           rocblas_int    lda,
                                           rocblas_stride stride_A,
                                           TConstPtr      x,
                                           rocblas_int    incx,
                                           rocblas_stride stride_x,
                                           TPtr           C,
                                           rocblas_int    ldc,
                                           rocblas_stride stride_c,
                                           rocblas_int    batch_count,
                                           const int      check_numerics,
                                           bool           is_input);
