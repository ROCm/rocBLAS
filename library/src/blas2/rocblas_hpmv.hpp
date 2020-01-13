/* ************************************************************************
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#ifndef __ROCBLAS_TBMV_HPP__
#define __ROCBLAS_TBMV_HPP__
#include "../blas1/rocblas_copy.hpp"
#include "handle.h"
#include "rocblas.h"
#include "utility.h"

/**
  *  A combined kernel to handle all hpmv cases.
  */
template <rocblas_int DIM_X, rocblas_int DIM_Y, typename T>
__device__ void hpmvx_kernel_calc(bool        upper,
                                  rocblas_int n,
                                  T           alpha,
                                  const T*    AP,
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
    T            res_A = 0.0;
    rocblas_int  col   = ty;

    for(col = ty; col < n; col += DIM_Y)
    {
        if(ind < n)
        {
            int  ind_x = ind;
            int  ind_y = col;
            bool CONJ  = false;

            if((ind > col && upper) || (ind < col && !upper))
            {
                // in the opposite triangule, get conjugate of value at transposed position
                ind_x = col;
                ind_y = ind;
                CONJ  = true;
            }

            // The indices used here for AP come from the summation of the number of elements
            // in previous columns.
            //                              col
            // For upper matrices, index = sigma(i) + row.
            //                              i=1
            //
            //                              col-1
            // For lower matrices, index = sigma(n-i) + row
            //                              i=0
            int index = upper ? ((ind_y * (ind_y + 1)) / 2) + ind_x
                              : ((ind_y * (2 * n - ind_y + 1)) / 2) + (ind_x - ind_y);

            res_A += (ind_x == ind_y ? std::real(AP[index]) : CONJ ? conj(AP[index]) : (AP[index]))
                     * x[col * incx];
        }
    }

    // Store partial sums for the diagonal
    sdata[tx + ty * DIM_X] = res_A;
    __syncthreads();

    thread_id = hipThreadIdx_x + hipThreadIdx_y * hipBlockDim_x;
    ind       = hipBlockIdx_x * DIM_X + thread_id;
    if(thread_id < DIM_X && ind < n)
    {
        // Add the partial sums of each diagonal and store
        for(rocblas_int i = 1; i < DIM_Y; i++)
        {
            sdata[thread_id] += sdata[thread_id + DIM_X * i];
        }

        // Update y.
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
  *  Loads pointers (in case of future batched versions) and launches
  *  the actual calculation kernel.
  *
  *  Summary of packed triangular matrices:
  *
  *  For Upper Triangular:
  *    The matrix is compacted so that AP contains A column-by-column, so that:
  *    AP(0) = A(0, 0),
  *    AP(1) = A(0, 1),
  *    AP(2) = A(1, 1), etc.
  *
  *    Ex: (rocblas_fill_upper; n = 4)
  *    1 2 4 7
  *    0 3 5 8
  *    0 0 6 9  ---->  (1, 2, 3, 4, 5, 6, 7, 8, 9, 8)
  *    0 0 0 8
  *
  *  For Lower Triangular:
  *    The matrix is compacted so that AP contains A column-by-column, so that:
  *    AP(0) = A(0, 0)
  *    AP(1) = A(1, 0)
  *    AP(2) = A(2, 0)
  *
  *    Ex: (rocblas_fill_lower; n = 4)
  *    1 0 0 0
  *    2 5 0 0
  *    3 6 8 0  ---->  (1, 2, 3, 4, 5, 6, 7, 8, 9, 8)
  *    4 7 9 8
  */
template <rocblas_int DIM_X, rocblas_int DIM_Y, typename U, typename V, typename W>
__global__ void hpmvx_kernel(bool           upper,
                             rocblas_int    n,
                             U              alphaa,
                             V              APa,
                             ptrdiff_t      shifta,
                             rocblas_stride strideA,
                             V              xa,
                             ptrdiff_t      shiftx,
                             rocblas_int    incx,
                             rocblas_stride stridex,
                             U              betaa,
                             W              ya,
                             ptrdiff_t      shifty,
                             rocblas_int    incy,
                             rocblas_stride stridey)
{
    rocblas_int num_threads = hipBlockDim_x * hipBlockDim_y * hipBlockDim_z;
    if(DIM_X * DIM_Y != num_threads)
        return; // need to launch exactly the same number of threads as template parameters indicate

    auto AP = load_ptr_batch(APa, hipBlockIdx_y, shifta, strideA);
    auto x  = load_ptr_batch(xa, hipBlockIdx_y, shiftx, stridex);
    auto y  = load_ptr_batch(ya, hipBlockIdx_y, shifty, stridey);

    auto alpha = load_scalar(alphaa);
    auto beta  = load_scalar(betaa);

    hpmvx_kernel_calc<DIM_X, DIM_Y>(upper, n, alpha, AP, x, incx, beta, y, incy);
}

/**
  *  U is always: const T* (either host or device)
  *  V is either: const T* OR const T* const*
  *  W is either:       T* OR       T* const*
  */
template <typename U, typename V, typename W>
rocblas_status rocblas_hpmv_template(rocblas_handle handle,
                                     rocblas_fill   uplo,
                                     rocblas_int    n,
                                     U              alpha,
                                     V              AP,
                                     rocblas_int    offseta,
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
    // quick return
    if(!n || !batch_count)
        return rocblas_status_success;

    // in case of negative inc shift pointer to end of data for negative indexing tid*inc
    offsetx = incx < 0 ? offsetx - ptrdiff_t(incx) * (n - 1) : offsetx;
    offsety = incy < 0 ? offsety - ptrdiff_t(incy) * (n - 1) : offsety;

    // (gemv) hpmvX_DIM_Y must be at least 4, 8 * 8 is very slow only 40Gflop/s
    static constexpr int hpmvX_DIM_X = 64;
    static constexpr int hpmvX_DIM_Y = 16;
    rocblas_int          blocks      = (n - 1) / (hpmvX_DIM_X) + 1;
    dim3                 hpmvx_grid(blocks, batch_count);
    dim3                 hpmvx_threads(hpmvX_DIM_X, hpmvX_DIM_Y);

    // Launch a modified gemv kernel. The logic is similar to gemv just with modified
    // indices for the banded matrices.
    if(handle->pointer_mode == rocblas_pointer_mode_device)
    {
        hipLaunchKernelGGL((hpmvx_kernel<hpmvX_DIM_X, hpmvX_DIM_Y>),
                           hpmvx_grid,
                           hpmvx_threads,
                           0,
                           handle->rocblas_stream,
                           uplo == rocblas_fill_upper,
                           n,
                           alpha,
                           AP,
                           offseta,
                           strideA,
                           x,
                           offsetx,
                           incx,
                           stridex,
                           beta,
                           y,
                           offsety,
                           incy,
                           stridey);
    }
    else
    {
        if(!*alpha && *beta == 1)
            return rocblas_status_success;

        hipLaunchKernelGGL((hpmvx_kernel<hpmvX_DIM_X, hpmvX_DIM_Y>),
                           hpmvx_grid,
                           hpmvx_threads,
                           0,
                           handle->rocblas_stream,
                           uplo == rocblas_fill_upper,
                           n,
                           *alpha,
                           AP,
                           offseta,
                           strideA,
                           x,
                           offsetx,
                           incx,
                           stridex,
                           *beta,
                           y,
                           offsety,
                           incy,
                           stridey);
    }

    return rocblas_status_success;
}

#endif
