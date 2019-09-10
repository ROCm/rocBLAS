/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#pragma once
#include "handle.h"
#include "rocblas.h"

template <typename T>
__global__ void rocblas_swap_kernel(rocblas_int n, T* x, rocblas_int incx, T* y, rocblas_int incy)
{
    ptrdiff_t tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if(tid < n)
    {
        auto tmp      = y[tid * incy];
        y[tid * incy] = x[tid * incx];
        x[tid * incx] = tmp;
    }
}

template <rocblas_int NB, typename T>
rocblas_status rocblas_swap_template(
    rocblas_handle handle, rocblas_int n, T* x, rocblas_int incx, T* y, rocblas_int incy)
{
    // Quick return if possible.
    if(n <= 0)
        return rocblas_status_success;

    hipStream_t rocblas_stream = handle->rocblas_stream;
    int         blocks         = (n - 1) / NB + 1;
    dim3        grid(blocks);
    dim3        threads(NB);

    if(incx < 0)
        x -= ptrdiff_t(incx) * (n - 1);
    if(incy < 0)
        y -= ptrdiff_t(incy) * (n - 1);

    hipLaunchKernelGGL(rocblas_swap_kernel, grid, threads, 0, rocblas_stream, n, x, incx, y, incy);

    return rocblas_status_success;
}
