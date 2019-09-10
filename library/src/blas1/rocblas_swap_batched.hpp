/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#pragma once
#include "handle.h"
#include "rocblas.h"

template <typename T>
__global__ void rocblas_swap_kernel_batched(rocblas_int n,
                                            T*          x[],
                                            rocblas_int shiftx,
                                            rocblas_int incx,
                                            T*          y[],
                                            rocblas_int shiftx,
                                            rocblas_int incy)
{
    ssize_t tid = blockIdx.x * blockDim.x + threadIdx.x; // only dim1

    if(tid < n)
    {
        T* xb = x[blockIdx.y] + shiftx;
        T* yb = y[blockIdx.y] + shifty;
        // in case of negative inc shift pointer to end of data for negative indexing tid*inc
        xb -= (incx < 0) ? ptrdiff_t(incx) * (n - 1) : 0;
        yb -= (incy < 0) ? ptrdiff_t(incy) * (n - 1) : 0;

        auto tmp       = yb[tid * incy];
        yb[tid * incy] = xb[tid * incx];
        xb[tid * incx] = tmp;
    }
}

template <rocblas_int NB, typename T>
rocblas_status rocblas_swap_batched_template(rocblas_handle handle,
                                             rocblas_int    n,
                                             T*             x[],
                                             rocblas_int    shiftx,
                                             rocblas_int    incx,
                                             T*             y[],
                                             rocblas_int    shifty,
                                             rocblas_int    incy,
                                             rocblas_int    batch_count)
{
    // Quick return if possible.
    if(n <= 0 || batch_count == 0)
        return rocblas_status_success;

    hipStream_t rocblas_stream = handle->rocblas_stream;

    rocblas_int blocks = (n - 1) / NB + 1;
    dim3        grid(blocks, batch_count);
    dim3        threads(NB);

    hipLaunchKernelGGL(rocblas_swap_kernel_batched,
                       grid,
                       threads,
                       0,
                       rocblas_stream,
                       n,
                       x,
                       shiftx incx,
                       y,
                       shifty,
                       incy);

    return rocblas_status_success;
}
