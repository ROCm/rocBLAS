/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#pragma once
#include "handle.h"

template <typename T, typename U>
__device__ void her_kernel_calc(
    bool upper, rocblas_int n, U alpha, const T* x, rocblas_int incx, T* A, rocblas_int lda)
{
    rocblas_int tx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    rocblas_int ty = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    if(upper ? ty < n && tx < ty : tx < n && ty < tx)
        A[tx + lda * ty] += alpha * x[tx * incx] * conj(x[ty * incx]);
    else if(tx == ty && tx < n)
    {
        U x_real = std::real(x[tx * incx]);
        U x_imag = std::imag(x[tx * incx]);
        A[tx + lda * ty]
            = std::real(A[tx + lda * ty]) + alpha * ((x_real * x_real) + (x_imag * x_imag));
    }
}

template <rocblas_int DIM_X, rocblas_int DIM_Y, typename TScal, typename TConstPtr, typename TPtr>
__global__ void rocblas_her_kernel(bool           upper,
                                   rocblas_int    n,
                                   TScal          alphaa,
                                   TConstPtr      xa,
                                   ptrdiff_t      shift_x,
                                   rocblas_int    incx,
                                   rocblas_stride stride_x,
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
    auto        alpha = load_scalar(alphaa);

    if(!alpha)
        return;

    her_kernel_calc(upper, n, alpha, x, incx, A, lda);
}

/**
 * TScal     is always: const U* (either host or device)
 * TConstPtr is either: const T* OR const T* const*
 * TPtr      is either:       T* OR       T* const*
 * Where T is the base type (rocblas_float_complex or rocblas_double_complex)
 * and U is the scalar type (float or double)
 */
template <typename TScal, typename TConstPtr, typename TPtr>
rocblas_status rocblas_her_template(rocblas_handle handle,
                                    rocblas_fill   uplo,
                                    rocblas_int    n,
                                    TScal          alpha,
                                    TConstPtr      x,
                                    rocblas_int    offset_x,
                                    rocblas_int    incx,
                                    rocblas_stride stride_x,
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

    static constexpr int HER_DIM_X = 128;
    static constexpr int HER_DIM_Y = 8;
    rocblas_int          blocksX   = (n - 1) / HER_DIM_X + 1;
    rocblas_int          blocksY   = (n - 1) / HER_DIM_Y + 1;

    dim3 her_grid(blocksX, blocksY, batch_count);
    dim3 her_threads(HER_DIM_X, HER_DIM_Y);

    if(rocblas_pointer_mode_device == handle->pointer_mode)
    {
        hipLaunchKernelGGL((rocblas_her_kernel<HER_DIM_X, HER_DIM_Y>),
                           her_grid,
                           her_threads,
                           0,
                           handle->rocblas_stream,
                           uplo == rocblas_fill_upper,
                           n,
                           alpha,
                           x,
                           shift_x,
                           incx,
                           stride_x,
                           A,
                           lda,
                           offset_A,
                           stride_A);
    }
    else
        hipLaunchKernelGGL((rocblas_her_kernel<HER_DIM_X, HER_DIM_Y>),
                           her_grid,
                           her_threads,
                           0,
                           handle->rocblas_stream,
                           uplo == rocblas_fill_upper,
                           n,
                           *alpha,
                           x,
                           shift_x,
                           incx,
                           stride_x,
                           A,
                           lda,
                           offset_A,
                           stride_A);

    return rocblas_status_success;
}
