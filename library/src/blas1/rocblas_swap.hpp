/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#pragma once
#include "handle.h"

template <typename T>
__forceinline__ __device__ __host__ void rocblas_swap_vals(T* x, T* y)
{
    T tmp = *y;
    *y    = *x;
    *x    = tmp;
}

template <typename UPtr>
__global__ void rocblas_swap_kernel(rocblas_int    n,
                                    UPtr           xa,
                                    ptrdiff_t      offsetx,
                                    rocblas_int    incx,
                                    rocblas_stride stridex,
                                    UPtr           ya,
                                    ptrdiff_t      offsety,
                                    rocblas_int    incy,
                                    rocblas_stride stridey)
{
    auto*     x   = load_ptr_batch(xa, hipBlockIdx_y, offsetx, stridex);
    auto*     y   = load_ptr_batch(ya, hipBlockIdx_y, offsety, stridey);
    ptrdiff_t tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(tid < n)
    {
        rocblas_swap_vals(x + tid * incx, y + tid * incy);
    }
}

template <rocblas_int NB, typename U>
rocblas_status rocblas_swap_template(rocblas_handle handle,
                                     rocblas_int    n,
                                     U              x,
                                     rocblas_int    offsetx,
                                     rocblas_int    incx,
                                     rocblas_stride stridex,
                                     U              y,
                                     rocblas_int    offsety,
                                     rocblas_int    incy,
                                     rocblas_stride stridey,
                                     rocblas_int    batch_count = 1)
{
    // Quick return if possible.
    if(n <= 0 || batch_count <= 0)
        return rocblas_status_success;

    dim3        blocks((n - 1) / NB + 1, batch_count);
    dim3        threads(NB);
    hipStream_t rocblas_stream = handle->rocblas_stream;

    // in case of negative inc shift pointer to end of data for negative indexing tid*inc
    ptrdiff_t shiftx = incx < 0 ? offsetx - ptrdiff_t(incx) * (n - 1) : offsetx;
    ptrdiff_t shifty = incy < 0 ? offsety - ptrdiff_t(incy) * (n - 1) : offsety;

    hipLaunchKernelGGL(rocblas_swap_kernel,
                       blocks,
                       threads,
                       0,
                       rocblas_stream,
                       n,
                       x,
                       shiftx,
                       incx,
                       stridex,
                       y,
                       shifty,
                       incy,
                       stridey);

    return rocblas_status_success;
}
