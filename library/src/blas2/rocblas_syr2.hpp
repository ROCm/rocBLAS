/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#pragma once
#include "handle.h"
#include "rocblas.h"

template <typename T>
__device__ void syr2_kernel_calc(bool        upper,
                                 rocblas_int n,
                                 T           alpha,
                                 const T*    x,
                                 rocblas_int incx,
                                 const T*    y,
                                 rocblas_int incy,
                                 T*          A,
                                 rocblas_int lda)
{
    rocblas_int tx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    rocblas_int ty = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    if(upper ? ty < n && tx <= ty : tx < n && ty <= tx)
        A[tx + ty * lda]
            += alpha * x[tx * incx] * y[ty * incy] + alpha * y[tx * incy] * x[ty * incx];
}

template <rocblas_int DIM_X, rocblas_int DIM_Y, typename TScal, typename TConstPtr, typename TPtr>
__global__ void rocblas_syr2_kernel(bool           upper,
                                    rocblas_int    n,
                                    TScal          alphaa,
                                    TConstPtr      xa,
                                    ptrdiff_t      shift_x,
                                    rocblas_int    incx,
                                    rocblas_stride stride_x,
                                    TConstPtr      ya,
                                    ptrdiff_t      shift_y,
                                    rocblas_int    incy,
                                    rocblas_stride stride_y,
                                    TPtr           Aa,
                                    rocblas_int    lda,
                                    ptrdiff_t      shift_A,
                                    rocblas_stride stride_A)
{
    rocblas_int num_threads = hipBlockDim_x * hipBlockDim_y * hipBlockDim_z;
    if(DIM_X * DIM_Y != num_threads)
        return; // need to launch exactly the number of threads as template parameters indicate.

    auto*       A     = load_ptr_batch(Aa, hipBlockIdx_z, shift_A, stride_A);
    const auto* x     = load_ptr_batch(xa, hipBlockIdx_z, shift_x, stride_x);
    const auto* y     = load_ptr_batch(ya, hipBlockIdx_z, shift_y, stride_y);
    auto        alpha = load_scalar(alphaa);

    if(!alpha)
        return;

    syr2_kernel_calc(upper, n, alpha, x, incx, y, incy, A, lda);
}

/**
 * TScal     is always: const T* (either host or device)
 * TConstPtr is either: const T* OR const T* const*
 * TPtr      is either:       T* OR       T* const*
 * Where T is the bast type (float or double)
 */
template <typename TScal, typename TConstPtr, typename TPtr>
ROCBLAS_EXPORT_NOINLINE rocblas_status rocblas_syr2_template(rocblas_handle handle,
                                                             rocblas_fill   uplo,
                                                             rocblas_int    n,
                                                             TScal          alpha,
                                                             TConstPtr      x,
                                                             rocblas_int    offset_x,
                                                             rocblas_int    incx,
                                                             rocblas_stride stride_x,
                                                             TConstPtr      y,
                                                             rocblas_int    offset_y,
                                                             rocblas_int    incy,
                                                             rocblas_stride stride_y,
                                                             TPtr           A,
                                                             rocblas_int    lda,
                                                             rocblas_int    offset_A,
                                                             rocblas_stride stride_A,
                                                             rocblas_int    batch_count)
{
    // Quick return if possible. Not Argument error
    if(!n || !batch_count)
        return rocblas_status_success;

    // in case of negative inc, shift pointer to end of data for negative indexing tid*inc
    ptrdiff_t shift_x = incx < 0 ? offset_x - ptrdiff_t(incx) * (n - 1) : offset_x;
    ptrdiff_t shift_y = incy < 0 ? offset_y - ptrdiff_t(incy) * (n - 1) : offset_y;

    static constexpr int SYR2_DIM_X = 128;
    static constexpr int SYR2_DIM_Y = 8;
    rocblas_int          blocksX    = (n - 1) / SYR2_DIM_X + 1;
    rocblas_int          blocksY    = (n - 1) / SYR2_DIM_Y + 1;

    dim3 syr2_grid(blocksX, blocksY, batch_count);
    dim3 syr2_threads(SYR2_DIM_X, SYR2_DIM_Y);

    if(rocblas_pointer_mode_device == handle->pointer_mode)
        hipLaunchKernelGGL((rocblas_syr2_kernel<SYR2_DIM_X, SYR2_DIM_Y>),
                           syr2_grid,
                           syr2_threads,
                           0,
                           handle->rocblas_stream,
                           uplo == rocblas_fill_upper,
                           n,
                           alpha,
                           x,
                           shift_x,
                           incx,
                           stride_x,
                           y,
                           shift_y,
                           incy,
                           stride_y,
                           A,
                           lda,
                           offset_A,
                           stride_A);
    else
        hipLaunchKernelGGL((rocblas_syr2_kernel<SYR2_DIM_X, SYR2_DIM_Y>),
                           syr2_grid,
                           syr2_threads,
                           0,
                           handle->rocblas_stream,
                           uplo == rocblas_fill_upper,
                           n,
                           *alpha,
                           x,
                           shift_x,
                           incx,
                           stride_x,
                           y,
                           shift_y,
                           incy,
                           stride_y,
                           A,
                           lda,
                           offset_A,
                           stride_A);

    return rocblas_status_success;
}
