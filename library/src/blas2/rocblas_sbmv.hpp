/* ************************************************************************
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#pragma once
#include "handle.h"

/**
  *  create partial sums for each ty.
  */
template <bool UPPER, rocblas_int DIM_Y, typename T>
inline __device__ T sbmv_kernel_helper(rocblas_int ty,
                                       rocblas_int ind,
                                       rocblas_int n,
                                       rocblas_int k,
                                       const T* __restrict__ A,
                                       rocblas_int lda,
                                       const T* __restrict__ x,
                                       rocblas_int incx)
{
    T           res_A = 0.0;
    rocblas_int col   = ty; // ty defines the column of banded matrix

    // Since the column is consistent, we iterate up diagonally in banded format
    for(col = ty; col < n; col += DIM_Y)
    {
        // We have to convert ind to banded matrix row
        rocblas_int row = UPPER ? ind + (k - col) : ind - col;

        if(ind < n)
        {
            if((ind <= col && UPPER) || (ind >= col && !UPPER))
            {
                // in upper/lower triangular part
                if(row <= k && row >= 0)
                {
                    res_A += A[row + col * lda] * x[col * incx];
                }
            }
            else
            {
                // in the opposite triangle, get value at transposed position
                rocblas_int trans_row = col;
                rocblas_int trans_col = ind;
                trans_row             = UPPER ? trans_row + (k - trans_col) : trans_row - trans_col;
                if(trans_row <= k && trans_row >= 0)
                {
                    res_A += A[trans_row + trans_col * lda] * x[col * incx];
                }
            }
        }
    }
    return res_A;
}

/**
  *  Computes y := alpha*A*x + beta*y where A is a symmetric matrix.
  *  If uplo == upper, the strictly lower part of A is not referenced,
  *  if uplo == lower, the strictly upper part of A is not referenced.
  */
template <bool UPPER, rocblas_int DIM_X, rocblas_int DIM_Y, typename T>
inline __device__ void sbmv_kernel_calc(rocblas_int n,
                                        rocblas_int k,
                                        T           alpha,
                                        const T* __restrict__ A,
                                        rocblas_int lda,
                                        const T* __restrict__ x,
                                        rocblas_int incx,
                                        T           beta,
                                        T* __restrict__ y,
                                        rocblas_int incy)
{
    rocblas_int thread_id = hipThreadIdx_x + hipThreadIdx_y * hipBlockDim_x;

    // threads are all configurated locally
    rocblas_int tx = thread_id % DIM_X;
    rocblas_int ty = thread_id / DIM_X;

    rocblas_int ind = hipBlockIdx_x * DIM_X + tx;

    __shared__ T sdata[DIM_X * DIM_Y];

    T res_A = 0.0;

    res_A = sbmv_kernel_helper<UPPER, DIM_Y>(ty, ind, n, k, A, lda, x, incx);

    sdata[tx + ty * DIM_X] = res_A;

    __syncthreads();

    ind = hipBlockIdx_x * DIM_X + thread_id;
    if(thread_id < DIM_X)
    {
        for(rocblas_int i = 1; i < DIM_Y; i++)
            sdata[thread_id] += sdata[thread_id + DIM_X * i];

        if(ind < n)
        {
            if(beta != 0)
            {
                y[ind * incy] = (alpha * sdata[thread_id]) + (beta * y[ind * incy]);
            }
            else
            {
                y[ind * incy] = alpha * sdata[thread_id];
            }
        }
    }
}

/**
  *  U is either: const T* OR T
  *  V is either: const T* OR const T* const*
  *  W is either:       T* OR       T* const*
  */
template <bool UPPER, rocblas_int DIM_X, rocblas_int DIM_Y, typename U, typename V, typename W>
__global__ void sbmv_kernel(rocblas_int    n,
                            rocblas_int    k,
                            U              alpha_device_host,
                            rocblas_stride stride_alpha,
                            V              Aa,
                            ptrdiff_t      shifta,
                            rocblas_int    lda,
                            rocblas_stride strideA,
                            V              xa,
                            ptrdiff_t      shiftx,
                            rocblas_int    incx,
                            rocblas_stride stridex,
                            U              beta_device_host,
                            rocblas_stride stride_beta,
                            W              ya,
                            ptrdiff_t      shifty,
                            rocblas_int    incy,
                            rocblas_stride stridey)
{
    rocblas_int num_threads = hipBlockDim_x * hipBlockDim_y * hipBlockDim_z;
    if(DIM_X * DIM_Y != num_threads)
        return; // need to launch exactly the same number of threads as template parameters indicate

    const auto* A = load_ptr_batch(Aa, hipBlockIdx_y, shifta, strideA);
    const auto* x = load_ptr_batch(xa, hipBlockIdx_y, shiftx, stridex);
    auto*       y = load_ptr_batch(ya, hipBlockIdx_y, shifty, stridey);

    auto alpha = load_scalar(alpha_device_host, hipBlockIdx_y, stride_alpha);
    auto beta  = load_scalar(beta_device_host, hipBlockIdx_y, stride_beta);

    sbmv_kernel_calc<UPPER, DIM_X, DIM_Y>(n, k, alpha, A, lda, x, incx, beta, y, incy);
}

template <typename T, typename U, typename V, typename W>
inline rocblas_status rocblas_sbmv_arg_check(rocblas_handle handle,
                                             rocblas_fill   uplo,
                                             rocblas_int    n,
                                             rocblas_int    k,
                                             const V*       alpha,
                                             rocblas_stride stride_alpha,
                                             const U*       A,
                                             rocblas_int    offseta,
                                             rocblas_int    lda,
                                             rocblas_stride strideA,
                                             const U*       x,
                                             rocblas_int    offsetx,
                                             rocblas_int    incx,
                                             rocblas_stride stridex,
                                             const V*       beta,
                                             rocblas_stride stride_beta,
                                             W*             y,
                                             rocblas_int    offsety,
                                             rocblas_int    incy,
                                             rocblas_stride stridey,
                                             rocblas_int    batch_count)
{
    // only supports stride_alpha and stride_beta for device memory alpha/beta
    if((handle->pointer_mode == rocblas_pointer_mode_host) && (stride_alpha || stride_beta))
        return rocblas_status_not_implemented;

    if(uplo != rocblas_fill_lower && uplo != rocblas_fill_upper)
        return rocblas_status_invalid_value;

    if(n < 0 || k < 0 || lda < k + 1 || lda < 1 || !incx || !incy || batch_count < 0)
        return rocblas_status_invalid_size;

    // quick return before pointer checks
    if(!n || !batch_count)
        return rocblas_status_success;

    if(!A || !x || !y || !alpha || !beta)
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <typename T, typename U, typename V, typename W>
rocblas_status rocblas_sbmv_template(rocblas_handle handle,
                                     rocblas_fill   uplo,
                                     rocblas_int    n,
                                     rocblas_int    k,
                                     const V*       alpha,
                                     rocblas_stride stride_alpha,
                                     const U*       A,
                                     rocblas_int    offseta,
                                     rocblas_int    lda,
                                     rocblas_stride strideA,
                                     const U*       x,
                                     rocblas_int    offsetx,
                                     rocblas_int    incx,
                                     rocblas_stride stridex,
                                     const V*       beta,
                                     rocblas_stride stride_beta,
                                     W*             y,
                                     rocblas_int    offsety,
                                     rocblas_int    incy,
                                     rocblas_stride stridey,
                                     rocblas_int    batch_count)
{
    //quick return
    if(!n || !batch_count)
        return rocblas_status_success;

    hipStream_t rocblas_stream = handle->rocblas_stream;

    // in case of negative inc shift pointer to end of data for negative indexing tid*inc
    auto shiftx = incx < 0 ? offsetx - ptrdiff_t(incx) * (n - 1) : offsetx;
    auto shifty = incy < 0 ? offsety - ptrdiff_t(incy) * (n - 1) : offsety;

    static constexpr int sbmv_DIM_X = 64;
    static constexpr int sbmv_DIM_Y = 16;
    rocblas_int          blocks     = (n - 1) / (sbmv_DIM_X) + 1;
    dim3                 grid(blocks, batch_count);
    dim3                 threads(sbmv_DIM_X, sbmv_DIM_Y);

    if(handle->pointer_mode == rocblas_pointer_mode_device)
    {
        if(uplo == rocblas_fill_upper)
        {
            hipLaunchKernelGGL((sbmv_kernel<true, sbmv_DIM_X, sbmv_DIM_Y>),
                               grid,
                               threads,
                               0,
                               rocblas_stream,
                               n,
                               k,
                               alpha,
                               stride_alpha,
                               A,
                               offseta,
                               lda,
                               strideA,
                               x,
                               shiftx,
                               incx,
                               stridex,
                               beta,
                               stride_beta,
                               y,
                               shifty,
                               incy,
                               stridey);
        }
        else
        {
            hipLaunchKernelGGL((sbmv_kernel<false, sbmv_DIM_X, sbmv_DIM_Y>),
                               grid,
                               threads,
                               0,
                               rocblas_stream,
                               n,
                               k,
                               alpha,
                               stride_alpha,
                               A,
                               offseta,
                               lda,
                               strideA,
                               x,
                               shiftx,
                               incx,
                               stridex,
                               beta,
                               stride_beta,
                               y,
                               shifty,
                               incy,
                               stridey);
        }
    }
    else
    {
        // quick return only for non-batched
        if(batch_count == 1 && !*alpha && *beta == 1)
            return rocblas_status_success;

        if(uplo == rocblas_fill_upper)
        {
            hipLaunchKernelGGL((sbmv_kernel<true, sbmv_DIM_X, sbmv_DIM_Y>),
                               grid,
                               threads,
                               0,
                               rocblas_stream,
                               n,
                               k,
                               *alpha,
                               stride_alpha,
                               A,
                               offseta,
                               lda,
                               strideA,
                               x,
                               shiftx,
                               incx,
                               stridex,
                               *beta,
                               stride_beta,
                               y,
                               shifty,
                               incy,
                               stridey);
        }
        else
        {
            hipLaunchKernelGGL((sbmv_kernel<false, sbmv_DIM_X, sbmv_DIM_Y>),
                               grid,
                               threads,
                               0,
                               rocblas_stream,
                               n,
                               k,
                               *alpha,
                               stride_alpha,
                               A,
                               offseta,
                               lda,
                               strideA,
                               x,
                               shiftx,
                               incx,
                               stridex,
                               *beta,
                               stride_beta,
                               y,
                               shifty,
                               incy,
                               stridey);
        }
    }

    return rocblas_status_success;
}
