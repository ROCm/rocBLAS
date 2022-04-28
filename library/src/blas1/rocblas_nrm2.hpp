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

#include "rocblas_reduction_template.hpp"

template <typename T, std::enable_if_t<!std::is_same<T, rocblas_half>{}, int> = 0>
__device__ __host__ inline auto fetch_abs2(T A)
{
    return std::norm(A);
}

template <typename T, std::enable_if_t<std::is_same<T, rocblas_half>{}, int> = 0>
__device__ __host__ inline auto fetch_abs2(T A)
{
    return A * A;
}
template <class To>
struct rocblas_fetch_nrm2
{
    template <class Ti>
    __forceinline__ __device__ To operator()(Ti x) const
    {
        return {fetch_abs2(x)};
    }
};

struct rocblas_finalize_nrm2
{
    template <class To>
    __forceinline__ __host__ __device__ To operator()(To x) const
    {
        return sqrt(x);
    }
};

template <rocblas_int NB, bool ISBATCHED, typename Ti, typename To, typename Tex = To>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_nrm2_template(rocblas_handle handle,
                                   rocblas_int    n,
                                   const Ti*      x,
                                   rocblas_stride shiftx,
                                   rocblas_int    incx,
                                   rocblas_stride stridex,
                                   rocblas_int    batch_count,
                                   To*            results,
                                   Tex*           workspace)
{
    return rocblas_reduction_template<NB,
                                      ISBATCHED,
                                      rocblas_fetch_nrm2<To>,
                                      rocblas_reduce_sum,
                                      rocblas_finalize_nrm2>(
        handle, n, x, shiftx, incx, stridex, batch_count, results, workspace);
}
