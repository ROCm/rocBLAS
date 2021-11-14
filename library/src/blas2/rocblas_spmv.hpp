/* ************************************************************************
 * Copyright 2019-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "check_numerics_vector.hpp"
#include "handle.hpp"
#include "rocblas/rocblas.h"

/**
  *  Computes y := alpha*A*x + beta*y where A is a symmetric matrix.
  *  If uplo == upper, the strictly lower part of A is not referenced,
  *  if uplo == lower, the strictly upper part of A is not referenced.
  */
template <rocblas_int DIM_X, rocblas_int DIM_Y, typename T>
__device__ void spmv_kernel_calc(bool        upper,
                                 rocblas_int n,
                                 T           alpha,
                                 const T* __restrict__ AP,
                                 const T* __restrict__ x,
                                 rocblas_int incx,
                                 T           beta,
                                 T* __restrict__ y,
                                 rocblas_int incy)
{
    rocblas_int thread_id = hipThreadIdx_x + hipThreadIdx_y * hipBlockDim_x;

    if(!alpha)
    {
        rocblas_int ind = hipBlockIdx_x * DIM_X + thread_id;
        if(thread_id < DIM_X && ind < n)
        {
            y[ind * incy] = beta ? (beta * y[ind * incy]) : 0;
        }
        return;
    }

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
            int ind_x = ind;
            int ind_y = col;

            if((ind > col && upper) || (ind < col && !upper))
            {
                // in the opposite triangle, get transposed position
                ind_x = col;
                ind_y = ind;
            }

            // row, col to packed index
            int index = upper ? ((ind_y * (ind_y + 1)) / 2) + ind_x
                              : ((ind_y * (2 * n - ind_y + 1)) / 2) + (ind_x - ind_y);

            res_A += AP[index] * x[col * incx];
        }
    }

    // Store partial sums
    sdata[tx + ty * DIM_X] = res_A;

    __syncthreads();

    ind = hipBlockIdx_x * DIM_X + thread_id;
    if(thread_id < DIM_X && ind < n)
    {
        // Add the partial sums and store
        for(rocblas_int i = 1; i < DIM_Y; i++)
        {
            sdata[thread_id] += sdata[thread_id + DIM_X * i];
        }

        y[ind * incy]
            = beta ? (alpha * sdata[thread_id]) + (beta * y[ind * incy]) : alpha * sdata[thread_id];
    }
}

/**
  *  Loads pointers and launches the actual calculation kernel.
  */
template <rocblas_int DIM_X, rocblas_int DIM_Y, typename TScal, typename TConstPtr, typename TPtr>
ROCBLAS_KERNEL __launch_bounds__(DIM_X* DIM_Y) void spmv_kernel(bool           upper,
                                                                rocblas_int    n,
                                                                TScal          alpha_device_host,
                                                                rocblas_stride stride_alpha,
                                                                TConstPtr __restrict__ APa,
                                                                ptrdiff_t      shifta,
                                                                rocblas_stride strideA,
                                                                TConstPtr __restrict__ xa,
                                                                ptrdiff_t      shiftx,
                                                                rocblas_int    incx,
                                                                rocblas_stride stridex,
                                                                TScal          beta_device_host,
                                                                rocblas_stride stride_beta,
                                                                TPtr __restrict__ ya,
                                                                ptrdiff_t      shifty,
                                                                rocblas_int    incy,
                                                                rocblas_stride stridey)
{
    rocblas_int num_threads = hipBlockDim_x * hipBlockDim_y * hipBlockDim_z;
    if(DIM_X * DIM_Y != num_threads)
        return; // need to launch exactly the same number of threads as template parameters indicate

    auto alpha = load_scalar(alpha_device_host, hipBlockIdx_y, stride_alpha);
    auto beta  = load_scalar(beta_device_host, hipBlockIdx_y, stride_beta);
    if(!alpha && beta == 1)
        return;

    auto AP = cond_load_ptr_batch(alpha, APa, hipBlockIdx_y, shifta, strideA);
    auto x  = cond_load_ptr_batch(alpha, xa, hipBlockIdx_y, shiftx, stridex);

    auto y = load_ptr_batch(ya, hipBlockIdx_y, shifty, stridey);

    spmv_kernel_calc<DIM_X, DIM_Y>(upper, n, alpha, AP, x, incx, beta, y, incy);
}

/**
  *  match rocblas_spmv_template parameters for easy calling
*/
template <typename T, typename U, typename V, typename W>
inline rocblas_status rocblas_spmv_arg_check(rocblas_handle handle,
                                             rocblas_fill   uplo,
                                             rocblas_int    n,
                                             const V*       alpha,
                                             rocblas_stride stride_alpha,
                                             const U*       A,
                                             rocblas_int    offseta,
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

    if(n < 0 || !incx || !incy || batch_count < 0)
        return rocblas_status_invalid_size;

    if(!n || !batch_count)
        return rocblas_status_success;

    if(!A || !x || !y || !alpha || !beta)
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <typename T, typename U, typename V, typename W>
rocblas_status rocblas_spmv_template(rocblas_handle handle,
                                     rocblas_fill   uplo,
                                     rocblas_int    n,
                                     const V*       alpha,
                                     rocblas_stride stride_alpha,
                                     const U*       A,
                                     rocblas_int    offseta,
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

    hipStream_t rocblas_stream = handle->get_stream();

    // in case of negative inc shift pointer to end of data for negative indexing tid*inc
    auto shiftx = incx < 0 ? offsetx - ptrdiff_t(incx) * (n - 1) : offsetx;
    auto shifty = incy < 0 ? offsety - ptrdiff_t(incy) * (n - 1) : offsety;

    static constexpr int spmv_DIM_X = 64;
    static constexpr int spmv_DIM_Y = 16;
    rocblas_int          blocks     = (n - 1) / (spmv_DIM_X) + 1;
    dim3                 grid(blocks, batch_count);
    dim3                 threads(spmv_DIM_X, spmv_DIM_Y);

    bool upper = uplo == rocblas_fill_upper;
    if(handle->pointer_mode == rocblas_pointer_mode_device)
    {
        hipLaunchKernelGGL((spmv_kernel<spmv_DIM_X, spmv_DIM_Y>),
                           grid,
                           threads,
                           0,
                           rocblas_stream,
                           upper,
                           n,
                           alpha,
                           stride_alpha,
                           A,
                           offseta,
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
        // quick return only for non-batched
        if(batch_count == 1 && !*alpha && *beta == 1)
            return rocblas_status_success;

        hipLaunchKernelGGL((spmv_kernel<spmv_DIM_X, spmv_DIM_Y>),
                           grid,
                           threads,
                           0,
                           rocblas_stream,
                           upper,
                           n,
                           *alpha,
                           stride_alpha,
                           A,
                           offseta,
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

    return rocblas_status_success;
}

//TODO :-Add rocblas_check_numerics_sp_matrix_template for checking Matrix `A` which is a Symmetric Packed Matrix
template <typename T, typename U>
rocblas_status rocblas_spmv_check_numerics(const char*    function_name,
                                           rocblas_handle handle,
                                           rocblas_int    n,
                                           T              A,
                                           rocblas_int    offset_a,
                                           rocblas_stride stride_a,
                                           T              x,
                                           rocblas_int    offset_x,
                                           rocblas_int    inc_x,
                                           rocblas_stride stride_x,
                                           U              y,
                                           rocblas_int    offset_y,
                                           rocblas_int    inc_y,
                                           rocblas_stride stride_y,
                                           rocblas_int    batch_count,
                                           const int      check_numerics,
                                           bool           is_input)
{
    rocblas_status check_numerics_status
        = rocblas_internal_check_numerics_vector_template(function_name,
                                                          handle,
                                                          n,
                                                          x,
                                                          offset_x,
                                                          inc_x,
                                                          stride_x,
                                                          batch_count,
                                                          check_numerics,
                                                          is_input);
    if(check_numerics_status != rocblas_status_success)
        return check_numerics_status;

    check_numerics_status = rocblas_internal_check_numerics_vector_template(function_name,
                                                                            handle,
                                                                            n,
                                                                            y,
                                                                            offset_y,
                                                                            inc_y,
                                                                            stride_y,
                                                                            batch_count,
                                                                            check_numerics,
                                                                            is_input);

    return check_numerics_status;
}
