/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#pragma once
#include "handle.h"
#include "rocblas.h"
#include "rocblas_swap.hpp"

template <typename T>
__global__ void rocblas_swap_kernel_strided_batched(rocblas_int n,
                                                    T*          x,
                                                    rocblas_int shiftx,
                                                    rocblas_int incx,
                                                    rocblas_int stridex,
                                                    T*          y,
                                                    rocblas_int shifty,
                                                    rocblas_int incy,
                                                    rocblas_int stridey)
{
    ssize_t tid = blockIdx.x * blockDim.x + threadIdx.x; // only dim1

    if(tid < n)
    {
        T* xb = x + blockIdx.y * stridex + shiftx;
        T* yb = y + blockIdx.y * stridey + shifty;
        // in case of negative inc shift pointer to end of data for negative indexing tid*inc
        xb -= (incx < 0) ? ptrdiff_t(incx) * (n - 1) : 0;
        yb -= (incy < 0) ? ptrdiff_t(incy) * (n - 1) : 0;

        rocblas_swap_vals(xb + tid * incx, yb + tid * incy);
    }
}

template <rocblas_int NB, typename T>
rocblas_status rocblas_swap_strided_batched_template(rocblas_handle handle,
                                             rocblas_int    n,
                                             T*             x,
                                             rocblas_int    shiftx,
                                             rocblas_int    incx,
                                             rocblas_int    stridex,
                                             T*             y,
                                             rocblas_int    shifty,
                                             rocblas_int    incy,
                                             rocblas_int    stridey,
                                             rocblas_int    batch_count)
{
    // Quick return if possible.
    if(n <= 0 || batch_count == 0)
        return rocblas_status_success;

    hipStream_t rocblas_stream = handle->rocblas_stream;
    rocblas_int blocks         = (n - 1) / NB + 1;
    dim3        grid(blocks, batch_count);
    dim3        threads(NB);

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
