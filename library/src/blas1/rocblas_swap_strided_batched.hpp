/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#pragma once
#include "handle.h"
#include "rocblas.h"
#include "rocblas_swap.hpp"

template <typename T>
__global__ void rocblas_swap_kernel_strided_batched(rocblas_int    n,
                                                    T*             x,
                                                    ptrdiff_t      shiftx,
                                                    rocblas_int    incx,
                                                    rocblas_stride stridex,
                                                    T*             y,
                                                    ptrdiff_t      shifty,
                                                    rocblas_int    incy,
                                                    rocblas_stride stridey)
{
    ssize_t tid = blockIdx.x * blockDim.x + threadIdx.x; // only dim1

    if(tid < n)
    {
        T* xb = x + blockIdx.y * stridex + shiftx;
        T* yb = y + blockIdx.y * stridey + shifty;

        rocblas_swap_vals(xb + tid * incx, yb + tid * incy);
    }
}

template <rocblas_int NB, typename T>
rocblas_status rocblas_swap_strided_batched_template(rocblas_handle handle,
                                                     rocblas_int    n,
                                                     T*             x,
                                                     rocblas_int    offsetx,
                                                     rocblas_int    incx,
                                                     rocblas_stride stridex,
                                                     T*             y,
                                                     rocblas_int    offsety,
                                                     rocblas_int    incy,
                                                     rocblas_stride stridey,
                                                     rocblas_int    batch_count)
{
    // Quick return if possible.
    if(n <= 0 || batch_count == 0)
        return rocblas_status_success;

    hipStream_t rocblas_stream = handle->rocblas_stream;
    rocblas_int blocks         = (n - 1) / NB + 1;
    dim3        grid(blocks, batch_count);
    dim3        threads(NB);

    // in case of negative inc shift pointer to end of data for negative indexing tid*inc
    ptrdiff_t shiftx = offsetx - ((incx < 0) ? ptrdiff_t(incx) * (n - 1) : 0);
    ptrdiff_t shifty = offsety - ((incy < 0) ? ptrdiff_t(incy) * (n - 1) : 0);

    hipLaunchKernelGGL(rocblas_swap_kernel_strided_batched,
                       grid,
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
