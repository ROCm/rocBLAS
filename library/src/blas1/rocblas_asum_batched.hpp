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

#include "rocblas_asum.hpp"
#include "rocblas_reduction_impl.hpp"

template <rocblas_int NB, typename U, typename To>
rocblas_status rocblas_asum_batched_template(rocblas_handle handle,
                                             rocblas_int    n,
                                             U              x,
                                             rocblas_stride shiftx,
                                             rocblas_int    incx,
                                             rocblas_int    batch_count,
                                             To*            workspace,
                                             To*            results)
{
    static constexpr bool           isbatched = true;
    static constexpr rocblas_stride stridex_0 = 0;
    return rocblas_reduction_template<NB,
                                      isbatched,
                                      rocblas_fetch_asum<To>,
                                      rocblas_reduce_sum,
                                      rocblas_finalize_identity>(
        handle, n, x, shiftx, incx, stridex_0, batch_count, results, workspace);
}
