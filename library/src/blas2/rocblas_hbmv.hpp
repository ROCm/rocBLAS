/* ************************************************************************
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#ifndef __ROCBLAS_hbmv_HPP__
#define __ROCBLAS_hbmv_HPP__
#include "handle.h"

/**
  *  Helper for the non-transpose case. Iterates through each diagonal
  *  and creates partial sums for each ty.
  */
template <rocblas_int DIM_Y, typename T>
__device__ T hbmvn_kernel_helper(rocblas_int ty,
                                 rocblas_int ind,
                                 bool        upper,
                                 rocblas_int m,
                                 rocblas_int k,
                                 const T*    A,
                                 rocblas_int lda,
                                 const T*    x,
                                 rocblas_int incx)
{
    T           res_A = 0.0;
    rocblas_int col   = ty; // ty defines the column of banded & regular matrix

    // Since the column is consistent, we can iterate up the diagonal
    for(col = ty; col < m; col += DIM_Y)
    {
        // We have to convert ind to banded matrix row
        rocblas_int row = upper ? ind + (k - col) : ind - col;

        if(ind < m)
        {
            if((ind <= col && upper) || (ind >= col && !upper))
            {
                // in upper/lower triangular part
                if(row < k && row > 0)
                {
                    // not on main diagonal, simply multiply
                    res_A += (A[row + col * lda] * x[col * incx]);
                }
                else if(row == 0)
                {
                    // If main diagonal, assume 0 imaginary part.
                    if(!upper || (k == 0 && upper))
                        res_A += (std::real(A[row + col * lda]) * x[col * incx]);
                    else
                        res_A += (A[row + col * lda] * x[col * incx]);
                }
                else if(row == k)
                {
                    // If main diagonal, assume 0 imaginary part.
                    if(upper)
                        res_A += (std::real(A[row + col * lda]) * x[col * incx]);
                    else
                        res_A += (A[row + col * lda] * x[col * incx]);
                }
            }
            else
            {
                // in the opposite triangle, get conjugate of value at transposed position
                rocblas_int trans_row = col;
                rocblas_int trans_col = ind;
                trans_row             = upper ? trans_row + (k - trans_col) : trans_row - trans_col;
                if(trans_row <= k && trans_row >= 0)
                {
                    res_A += (conj(A[trans_row + trans_col * lda]) * x[col * incx]);
                }
            }
        }
    }
    return res_A;
}

/**
  *  Computes y := alpha*A*x + beta*y where A is a hermetian matrix.
  *  If uplo == upper, the strictly lower part of A is not referenced,
  *  if uplo == lower, the strictly upper part of A is not referenced.
  *  The imaginary part of the main diagonal is assumed to always be == 0.
  */
template <rocblas_int DIM_X, rocblas_int DIM_Y, typename T>
__device__ void hbmvn_kernel_calc(bool        upper,
                                  rocblas_int n,
                                  rocblas_int k,
                                  T           alpha,
                                  const T*    A,
                                  rocblas_int lda,
                                  const T*    x,
                                  rocblas_int incx,
                                  T           beta,
                                  T*          y,
                                  rocblas_int incy)
{
    rocblas_int thread_id = hipThreadIdx_x + hipThreadIdx_y * hipBlockDim_x;

    // threads are all configurated locally
    rocblas_int tx = thread_id % DIM_X;
    rocblas_int ty = thread_id / DIM_X;

    rocblas_int ind = hipBlockIdx_x * DIM_X + tx;

    __shared__ T sdata[DIM_X * DIM_Y];

    T res_A = 0.0;

    res_A = hbmvn_kernel_helper<DIM_Y>(ty, ind, upper, n, k, A, lda, x, incx);

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
template <rocblas_int DIM_X, rocblas_int DIM_Y, typename U, typename V, typename W>
__global__ void hbmvn_kernel(bool           upper,
                             rocblas_int    n,
                             rocblas_int    k,
                             U              alpha_device_host,
                             V              Aa,
                             ptrdiff_t      shifta,
                             rocblas_int    lda,
                             rocblas_stride strideA,
                             V              xa,
                             ptrdiff_t      shiftx,
                             rocblas_int    incx,
                             rocblas_stride stridex,
                             U              beta_device_host,
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

    auto alpha = load_scalar(alpha_device_host);
    auto beta  = load_scalar(beta_device_host);

    hbmvn_kernel_calc<DIM_X, DIM_Y>(upper, n, k, alpha, A, lda, x, incx, beta, y, incy);
}

/**
  *  U is always: const T* (either host or device)
  *  V is either: const T* OR const T* const*
  *  W is either:       T* OR       T* const*
  */
template <typename U, typename V, typename W>
rocblas_status rocblas_hbmv_template(rocblas_handle handle,
                                     rocblas_fill   uplo,
                                     rocblas_int    n,
                                     rocblas_int    k,
                                     U              alpha,
                                     V              A,
                                     rocblas_int    offseta,
                                     rocblas_int    lda,
                                     rocblas_stride strideA,
                                     V              x,
                                     rocblas_int    offsetx,
                                     rocblas_int    incx,
                                     rocblas_stride stridex,
                                     U              beta,
                                     W              y,
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

    // hbmvN_DIM_Y must be at least 4, 8 * 8 is very slow only 40Gflop/s
    static constexpr int hbmvN_DIM_X = 64;
    static constexpr int hbmvN_DIM_Y = 16;
    rocblas_int          blocks      = (n - 1) / (hbmvN_DIM_X) + 1;
    dim3                 hbmvn_grid(blocks, batch_count);
    dim3                 hbmvn_threads(hbmvN_DIM_X, hbmvN_DIM_Y);

    if(handle->pointer_mode == rocblas_pointer_mode_device)
    {
        hipLaunchKernelGGL((hbmvn_kernel<hbmvN_DIM_X, hbmvN_DIM_Y>),
                           hbmvn_grid,
                           hbmvn_threads,
                           0,
                           rocblas_stream,
                           uplo == rocblas_fill_upper,
                           n,
                           k,
                           alpha,
                           A,
                           offseta,
                           lda,
                           strideA,
                           x,
                           shiftx,
                           incx,
                           stridex,
                           beta,
                           y,
                           shifty,
                           incy,
                           stridey);
    }
    else
    {
        if(!*alpha && *beta == 1)
            return rocblas_status_success;

        hipLaunchKernelGGL((hbmvn_kernel<hbmvN_DIM_X, hbmvN_DIM_Y>),
                           hbmvn_grid,
                           hbmvn_threads,
                           0,
                           rocblas_stream,
                           uplo == rocblas_fill_upper,
                           n,
                           k,
                           *alpha,
                           A,
                           offseta,
                           lda,
                           strideA,
                           x,
                           shiftx,
                           incx,
                           stridex,
                           *beta,
                           y,
                           shifty,
                           incy,
                           stridey);
    }

    return rocblas_status_success;
}

#endif
