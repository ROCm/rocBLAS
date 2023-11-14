/* ************************************************************************
 * Copyright (C) 2016-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include "handle.hpp"
#include "int64_helpers.hpp"
#include "rocblas.h"
#include "rocblas_swap.hpp"

template <typename T>
__forceinline__ __device__ __host__ void rocblas_swap_vals(T* __restrict__ x, T* __restrict__ y)
{
    T tmp = *y;
    *y    = *x;
    *x    = tmp;
}

template <typename API_INT, rocblas_int NB, typename UPtr>
ROCBLAS_KERNEL(NB)
rocblas_swap_kernel(rocblas_int    n,
                    UPtr           xa,
                    rocblas_stride offsetx,
                    API_INT        incx,
                    rocblas_stride stridex,
                    UPtr           ya,
                    rocblas_stride offsety,
                    API_INT        incy,
                    rocblas_stride stridey)
{
    auto*   x   = load_ptr_batch(xa, blockIdx.y, offsetx, stridex);
    auto*   y   = load_ptr_batch(ya, blockIdx.y, offsety, stridey);
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid < n)
    {
        rocblas_swap_vals(x + tid * incx, y + tid * incy);
    }
}

//! @brief Optimized kernel for the floating points.
//!
template <rocblas_int NB, typename UPtr>
ROCBLAS_KERNEL(NB)
rocblas_sswap_2_kernel(rocblas_int n,
                       UPtr __restrict__ xa,
                       rocblas_stride offsetx,
                       rocblas_stride stridex,
                       UPtr __restrict__ ya,
                       rocblas_stride offsety,
                       rocblas_stride stridey)
{
    int64_t tid = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    auto*   x   = load_ptr_batch(xa, blockIdx.y, offsetx, stridex);
    auto*   y   = load_ptr_batch(ya, blockIdx.y, offsety, stridey);
    if(tid < n - 1)
    {
        for(int j = 0; j < 2; ++j)
        {
            rocblas_swap_vals(x + tid + j, y + tid + j);
        }
    }
    if(n % 2 != 0 && tid == n - 1)
    {
        rocblas_swap_vals(x + tid, y + tid);
    }
}

template <typename API_INT, rocblas_int NB, typename T>
rocblas_status rocblas_internal_swap_launcher(rocblas_handle handle,
                                              API_INT        n,
                                              T              x,
                                              rocblas_stride offsetx,
                                              API_INT        incx,
                                              rocblas_stride stridex,
                                              T              y,
                                              rocblas_stride offsety,
                                              API_INT        incy,
                                              rocblas_stride stridey,
                                              API_INT        batch_count)
{
    // Quick return if possible.
    if(n <= 0 || batch_count <= 0)
        return rocblas_status_success;

    static constexpr bool using_rocblas_float
        = std::is_same_v<T, rocblas_float*> || std::is_same_v<T, rocblas_float* const*>;

    if(!using_rocblas_float || incx != 1 || incy != 1)
    {
        // in case of negative inc shift pointer to end of data for negative indexing tid*inc
        int64_t shiftx = incx < 0 ? offsetx - int64_t(incx) * (n - 1) : offsetx;
        int64_t shifty = incy < 0 ? offsety - int64_t(incy) * (n - 1) : offsety;

        dim3 blocks((n - 1) / NB + 1, batch_count);
        dim3 threads(NB);

        ROCBLAS_LAUNCH_KERNEL((rocblas_swap_kernel<API_INT, NB>),
                              blocks,
                              threads,
                              0,
                              handle->get_stream(),
                              n,
                              x,
                              shiftx,
                              incx,
                              stridex,
                              y,
                              shifty,
                              incy,
                              stridey);
    }
    else
    {
        // Kernel function for improving the performance of SSWAP when incx==1 and incy==1
        int64_t shiftx = offsetx - 0;
        int64_t shifty = offsety - 0;

        int  blocks = 1 + ((n - 1) / (NB * 2));
        dim3 grid(blocks, batch_count);
        dim3 threads(NB);

        ROCBLAS_LAUNCH_KERNEL((rocblas_sswap_2_kernel<NB>),
                              grid,
                              threads,
                              0,
                              handle->get_stream(),
                              n,
                              x,
                              shiftx,
                              stridex,
                              y,
                              shifty,
                              stridey);
    }
    return rocblas_status_success;
}
