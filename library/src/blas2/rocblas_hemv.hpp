/* ************************************************************************
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#ifndef __ROCBLAS_HEMV_HPP__
#define __ROCBLAS_HEMV_HPP__
#include "handle.h"

/**
  *  Computes y := alpha*A*x + beta*y where A is a hermetian matrix.
  *  If uplo == upper, the strictly lower part of A is not referenced,
  *  if uplo == lower, the strictly upper part of A is not referenced.
  *  The imaginary part of the main diagonal is assumed to always be == 0.
  */
template <rocblas_int DIM_X, rocblas_int DIM_Y, typename T>
__device__ void hemvn_kernel_calc(rocblas_fill uplo,
                                  rocblas_int  n,
                                  T            alpha,
                                  const T*     A,
                                  rocblas_int  lda,
                                  const T*     x,
                                  rocblas_int  incx,
                                  T            beta,
                                  T*           y,
                                  rocblas_int  incy)
{
    rocblas_int thread_id = hipThreadIdx_x + hipThreadIdx_y * hipBlockDim_x;
    bool        upper     = uplo == rocblas_fill_upper;

    // threads are all configurated locally
    rocblas_int tx = thread_id % DIM_X;
    rocblas_int ty = thread_id / DIM_X;

    rocblas_int ind = hipBlockIdx_x * DIM_X + tx;

    __shared__ T sdata[DIM_X * DIM_Y];

    T res_A;
    T tmp_A;

    res_A = tmp_A   = T(0);
    rocblas_int col = ty;

    for(col = ty; col < n; col += DIM_Y)
    {
        if(ind < n)
        {
            if(col > ind)
                tmp_A = upper ? A[ind + col * lda] : conj(A[col + ind * lda]);
            else if(col < ind)
                tmp_A = !upper ? A[ind + col * lda] : conj(A[col + ind * lda]);
            else if(col == ind)
                tmp_A = std::real(A[ind + col * lda]);

            res_A += tmp_A * x[(col)*incx];
        }
    }

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
__global__ void hemvn_kernel(rocblas_fill   uplo,
                             rocblas_int    n,
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

    hemvn_kernel_calc<DIM_X, DIM_Y>(uplo, n, alpha, A, lda, x, incx, beta, y, incy);
}

/**
  *  U is always: const T* (either host or device)
  *  V is either: const T* OR const T* const*
  *  W is either:       T* OR       T* const*
  */
template <typename U, typename V, typename W>
rocblas_status rocblas_hemv_template(rocblas_handle handle,
                                     rocblas_fill   uplo,
                                     rocblas_int    n,
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
    if(!n || batch_count < 0)
        return rocblas_status_success;

    hipStream_t rocblas_stream = handle->rocblas_stream;

    // in case of negative inc shift pointer to end of data for negative indexing tid*inc
    auto shiftx = incx < 0 ? offsetx - ptrdiff_t(incx) * (n - 1) : offsetx;
    auto shifty = incy < 0 ? offsety - ptrdiff_t(incy) * (n - 1) : offsety;

    // HEMVN_DIM_Y must be at least 4, 8 * 8 is very slow only 40Gflop/s
    static constexpr int HEMVN_DIM_X = 64;
    static constexpr int HEMVN_DIM_Y = 16;
    rocblas_int          blocks      = (n - 1) / (HEMVN_DIM_X) + 1;
    dim3                 hemvn_grid(blocks, batch_count);
    dim3                 hemvn_threads(HEMVN_DIM_X, HEMVN_DIM_Y);

    if(handle->pointer_mode == rocblas_pointer_mode_device)
    {
        hipLaunchKernelGGL((hemvn_kernel<HEMVN_DIM_X, HEMVN_DIM_Y>),
                           hemvn_grid,
                           hemvn_threads,
                           0,
                           rocblas_stream,
                           uplo,
                           n,
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

        hipLaunchKernelGGL((hemvn_kernel<HEMVN_DIM_X, HEMVN_DIM_Y>),
                           hemvn_grid,
                           hemvn_threads,
                           0,
                           rocblas_stream,
                           uplo,
                           n,
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
