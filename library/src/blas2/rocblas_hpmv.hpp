/* ************************************************************************
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#ifndef __ROCBLAS_TBMV_HPP__
#define __ROCBLAS_TBMV_HPP__
#include "../blas1/rocblas_copy.hpp"

/**
  *  A combined kernel to handle all hpmv cases.
  */
template <rocblas_int DIM_X, rocblas_int DIM_Y, typename T>
__device__ void hpmv_kernel_calc(bool        upper,
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
                // in the opposite triangle, get conjugate of value at transposed position
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
  *  Loads pointers and launches the actual calculation kernel.
  */
template <rocblas_int DIM_X, rocblas_int DIM_Y, typename TScal, typename TConstPtr, typename TPtr>
__global__ void hpmv_kernel(bool           upper,
                            rocblas_int    n,
                            TScal          alphaa,
                            TConstPtr      APa,
                            ptrdiff_t      shifta,
                            rocblas_stride strideA,
                            TConstPtr      xa,
                            ptrdiff_t      shiftx,
                            rocblas_int    incx,
                            rocblas_stride stridex,
                            TScal          betaa,
                            TPtr           ya,
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

    hpmv_kernel_calc<DIM_X, DIM_Y>(upper, n, alpha, AP, x, incx, beta, y, incy);
}

/**
  *  TScal     is always: const T* (either host or device)
  *  TConstPtr is either: const T* OR const T* const*
  *  TPtr      is either:       T* OR       T* const*
  */
template <typename TScal, typename TConstPtr, typename TPtr>
rocblas_status rocblas_hpmv_template(rocblas_handle handle,
                                     rocblas_fill   uplo,
                                     rocblas_int    n,
                                     TScal          alpha,
                                     TConstPtr      AP,
                                     rocblas_int    offseta,
                                     rocblas_stride strideA,
                                     TConstPtr      x,
                                     rocblas_int    offsetx,
                                     rocblas_int    incx,
                                     rocblas_stride stridex,
                                     TScal          beta,
                                     TPtr           y,
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

    static constexpr int HPMV_DIM_X = 64;
    static constexpr int HPMV_DIM_Y = 16;
    rocblas_int          blocks     = (n - 1) / (HPMV_DIM_X) + 1;
    dim3                 hpmv_grid(blocks, batch_count);
    dim3                 hpmv_threads(HPMV_DIM_X, HPMV_DIM_Y);

    // Launch a modified gemv kernel for hpmv.
    if(handle->pointer_mode == rocblas_pointer_mode_device)
    {
        hipLaunchKernelGGL((hpmv_kernel<HPMV_DIM_X, HPMV_DIM_Y>),
                           hpmv_grid,
                           hpmv_threads,
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

        hipLaunchKernelGGL((hpmv_kernel<HPMV_DIM_X, HPMV_DIM_Y>),
                           hpmv_grid,
                           hpmv_threads,
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
