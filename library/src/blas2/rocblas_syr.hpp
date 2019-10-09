/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#pragma once
#include "handle.h"
#include "rocblas.h"

template <typename T, typename U>
__global__ void rocblas_syr_kernel(rocblas_fill uplo,
                                   rocblas_int  n,
                                   U            alpha_device_host,
                                   const T* __restrict__ x,
                                   rocblas_int incx,
                                   T*          A,
                                   rocblas_int lda)
{
    auto        alpha = load_scalar(alpha_device_host);
    rocblas_int tx    = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    rocblas_int ty    = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    if(uplo == rocblas_fill_lower ? tx < n && ty <= tx : ty < n && tx <= ty)
        A[tx + lda * ty] += alpha * x[tx * incx] * x[ty * incx];
}

template <typename T>
rocblas_status rocblas_syr_template(rocblas_handle handle,
                                    rocblas_fill   uplo,
                                    rocblas_int    n,
                                    const T*       alpha,
                                    const T*       x,
                                    rocblas_int    incx,
                                    T*             A,
                                    rocblas_int    lda)
{
    // Quick return if possible. Not Argument error
    if(!n)
        return rocblas_status_success;

    hipStream_t rocblas_stream = handle->rocblas_stream;

    static constexpr int GEMV_DIM_X = 128;
    static constexpr int GEMV_DIM_Y = 8;
    rocblas_int          blocksX    = (n - 1) / GEMV_DIM_X + 1;
    rocblas_int          blocksY    = (n - 1) / GEMV_DIM_Y + 1;

    dim3 syr_grid(blocksX, blocksY);
    dim3 syr_threads(GEMV_DIM_X, GEMV_DIM_Y);

    if(incx < 0)
        x -= ptrdiff_t(incx) * (n - 1);

    if(rocblas_pointer_mode_device == handle->pointer_mode)
        hipLaunchKernelGGL(rocblas_syr_kernel,
                           syr_grid,
                           syr_threads,
                           0,
                           rocblas_stream,
                           uplo,
                           n,
                           alpha,
                           x,
                           incx,
                           A,
                           lda);
    else
        hipLaunchKernelGGL(rocblas_syr_kernel,
                           syr_grid,
                           syr_threads,
                           0,
                           rocblas_stream,
                           uplo,
                           n,
                           *alpha,
                           x,
                           incx,
                           A,
                           lda);

    return rocblas_status_success;
}
