/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#pragma once
#include "handle.h"
#include "rocblas.h"

template <typename T, typename U>
__global__ void rocblas_syr_strided_batched_kernel(rocblas_fill   uplo,
                                                   rocblas_int    n,
                                                   U              alpha_device_host,
                                                   rocblas_stride stride_alpha,
                                                   const T* __restrict__ xvec,
                                                   ptrdiff_t      shiftx,
                                                   rocblas_int    incx,
                                                   rocblas_stride stridex,
                                                   T*             Avec,
                                                   rocblas_int    shiftA,
                                                   rocblas_int    lda,
                                                   rocblas_stride strideA)
{
    auto        alpha = load_scalar(alpha_device_host, hipBlockIdx_z, stride_alpha);
    rocblas_int tx    = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    rocblas_int ty    = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    const T* x = xvec + hipBlockIdx_z * stridex + shiftx;
    T*       A = Avec + hipBlockIdx_z * strideA + shiftA;

    if(uplo == rocblas_fill_lower ? tx < n && ty <= tx : ty < n && tx <= ty)
        A[tx + lda * ty] += alpha * x[tx * incx] * x[ty * incx];
}

template <typename T>
rocblas_status rocblas_syr_strided_batched_template(rocblas_handle handle,
                                                    rocblas_fill   uplo,
                                                    rocblas_int    n,
                                                    const T*       alpha,
                                                    rocblas_stride stride_alpha,
                                                    const T*       x,
                                                    rocblas_int    offsetx,
                                                    rocblas_int    incx,
                                                    rocblas_stride stridex,
                                                    T*             A,
                                                    rocblas_int    shiftA,
                                                    rocblas_int    lda,
                                                    rocblas_stride strideA,
                                                    rocblas_int    batch_count)
{

    // Quick return if possible. Not Argument error
    if(!n || batch_count == 0)
        return rocblas_status_success;

    hipStream_t rocblas_stream = handle->rocblas_stream;

    static constexpr int GEMV_DIM_X = 128;
    static constexpr int GEMV_DIM_Y = 8;
    rocblas_int          blocksX    = (n - 1) / GEMV_DIM_X + 1;
    rocblas_int          blocksY    = (n - 1) / GEMV_DIM_Y + 1;

    dim3 syr_strided_batched_grid(blocksX, blocksY, batch_count);
    dim3 syr_strided_batched_threads(GEMV_DIM_X, GEMV_DIM_Y);

    // in case of negative inc shift to end of data for negative indexing tid*inc
    ptrdiff_t shiftx = offsetx - ((incx < 0) ? ptrdiff_t(incx) * (n - 1) : 0);

    if(rocblas_pointer_mode_device == handle->pointer_mode)
        hipLaunchKernelGGL(rocblas_syr_strided_batched_kernel,
                           syr_strided_batched_grid,
                           syr_strided_batched_threads,
                           0,
                           rocblas_stream,
                           uplo,
                           n,
                           alpha,
                           stride_alpha,
                           x,
                           shiftx,
                           incx,
                           stridex,
                           A,
                           shiftA,
                           lda,
                           strideA);
    else
        hipLaunchKernelGGL(rocblas_syr_strided_batched_kernel,
                           syr_strided_batched_grid,
                           syr_strided_batched_threads,
                           0,
                           rocblas_stream,
                           uplo,
                           n,
                           *alpha,
                           stride_alpha,
                           x,
                           shiftx,
                           incx,
                           stridex,
                           A,
                           shiftA,
                           lda,
                           strideA);

    return rocblas_status_success;
}
