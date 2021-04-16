/* ************************************************************************
 * Copyright 2019-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "../blas1/rocblas_copy.hpp"
#include "check_numerics_vector.hpp"

/**
  *  A combined kernel to handle all hpmv cases.
  */
template <rocblas_int DIM_X, rocblas_int DIM_Y, typename T>
__device__ void hpmv_kernel_calc(bool        upper,
                                 rocblas_int n,
                                 T           alpha,
                                 const T*    AP,
                                 const T*    x,
                                 ptrdiff_t   incx,
                                 T           beta,
                                 T*          y,
                                 ptrdiff_t   incy)
{
    rocblas_int thread_id = hipThreadIdx_x + hipThreadIdx_y * hipBlockDim_x;

    // threads are all configurated locally
    rocblas_int tx = thread_id % DIM_X;
    rocblas_int ty = thread_id / DIM_X;

    rocblas_int ind = hipBlockIdx_x * DIM_X + tx;

    if(!alpha)
    {
        if(thread_id < DIM_X && ind < n)
        {
            rocblas_int idx = hipBlockIdx_x * DIM_X + thread_id;
            if(idx < n)
                y[idx * incy] = beta ? beta * y[idx * incy] : 0;
        }
        return;
    }

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
            // clang-format off
            res_A += (ind_x == ind_y ? std::real(AP[index]) : CONJ ? conj(AP[index]) : (AP[index]))
                     * x[col * incx];
            // clang-format on
        }
    }

    // Store partial sums for the diagonal
    sdata[tx + ty * DIM_X] = res_A;
    __syncthreads();

    if(thread_id < DIM_X && ind < n)
    {
        // Add the partial sums of each diagonal and store
        for(rocblas_int i = 1; i < DIM_Y; i++)
            sdata[thread_id] += sdata[thread_id + DIM_X * i];

        rocblas_int idx = hipBlockIdx_x * DIM_X + thread_id;
        // Update y.
        if(idx < n)
            y[idx * incy]
                = beta ? alpha * sdata[thread_id] + beta * y[idx * incy] : alpha * sdata[thread_id];
    }
}

/**
  *  Loads pointers and launches the actual calculation kernel.
  */
template <rocblas_int DIM_X, rocblas_int DIM_Y, typename TScal, typename TConstPtr, typename TPtr>
__launch_bounds__(DIM_X* DIM_Y) __global__ void hpmv_kernel(bool           upper,
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

    auto alpha = load_scalar(alphaa);
    auto beta  = load_scalar(betaa);

    if(!alpha && beta == 1)
        return;

    auto AP = cond_load_ptr_batch(alpha, APa, hipBlockIdx_y, shifta, strideA);
    auto x  = cond_load_ptr_batch(alpha, xa, hipBlockIdx_y, shiftx, stridex);

    auto y = load_ptr_batch(ya, hipBlockIdx_y, shifty, stridey);

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
                           handle->get_stream(),
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
                           handle->get_stream(),
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

//TODO :-Add rocblas_check_numerics_hp_matrix_template for checking Matrix `AP` which is a Hermitian Packed matrix
template <typename T, typename U>
rocblas_status rocblas_hpmv_check_numerics(const char*    function_name,
                                           rocblas_handle handle,
                                           rocblas_int    n,
                                           T              AP,
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
