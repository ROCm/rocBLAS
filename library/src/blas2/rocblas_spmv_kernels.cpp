/* ************************************************************************
 * Copyright (C) 2019-2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */

#include "check_numerics_vector.hpp"
#include "handle.hpp"
#include "rocblas.h"
#include "rocblas_spmv.hpp"

/**
  *  Computes y := alpha*A*x + beta*y where A is a symmetric matrix.
  *  If uplo == upper, the strictly lower part of A is not referenced,
  *  if uplo == lower, the strictly upper part of A is not referenced.
  */
template <rocblas_int DIM_X, rocblas_int DIM_Y, typename T>
__device__ void rocblas_spmv_kernel_calc(bool        is_upper,
                                         rocblas_int n,
                                         T           alpha,
                                         const T* __restrict__ AP,
                                         const T* __restrict__ x,
                                         rocblas_int incx,
                                         T           beta,
                                         T* __restrict__ y,
                                         rocblas_int incy)
{
    rocblas_int thread_id = threadIdx.x + threadIdx.y * blockDim.x;

    if(!alpha)
    {
        rocblas_int ind = blockIdx.x * DIM_X + thread_id;
        if(thread_id < DIM_X && ind < n)
        {
            y[ind * int64_t(incy)] = beta ? (beta * y[ind * int64_t(incy)]) : 0;
        }
        return;
    }

    // threads are all configurated locally
    rocblas_int tx = thread_id % DIM_X;
    rocblas_int ty = thread_id / DIM_X;

    rocblas_int ind = blockIdx.x * DIM_X + tx;

    __shared__ T sdata[DIM_X * DIM_Y];
    T            res_A = 0.0;
    rocblas_int  col;

    for(col = ty; col < n; col += DIM_Y)
    {
        if(ind < n)
        {
            int ind_x = ind;
            int ind_y = col;

            if((ind > col && is_upper) || (ind < col && !is_upper))
            {
                // in the opposite triangle, get transposed position
                ind_x = col;
                ind_y = ind;
            }

            // row, col to packed index
            size_t index = is_upper ? ((ind_y * (size_t(ind_y) + 1)) / 2) + ind_x
                                    : ((ind_y * (2 * size_t(n) - ind_y + 1)) / 2) + (ind_x - ind_y);

            res_A += AP[index] * x[col * int64_t(incx)];
        }
    }

    // Store partial sums
    sdata[tx + ty * DIM_X] = res_A;

    __syncthreads();

    ind = blockIdx.x * DIM_X + thread_id;
    if(thread_id < DIM_X && ind < n)
    {
        // Add the partial sums and store
        for(rocblas_int i = 1; i < DIM_Y; i++)
        {
            sdata[thread_id] += sdata[thread_id + DIM_X * i];
        }

        y[ind * int64_t(incy)] = beta ? (alpha * sdata[thread_id]) + (beta * y[ind * int64_t(incy)])
                                      : alpha * sdata[thread_id];
    }
}

/**
  *  Loads pointers and launches the actual calculation kernel.
  */
template <rocblas_int DIM_X, rocblas_int DIM_Y, typename TScal, typename TConstPtr, typename TPtr>
ROCBLAS_KERNEL(DIM_X* DIM_Y)
rocblas_spmv_kernel(bool           is_upper,
                    rocblas_int    n,
                    TScal          alpha_device_host,
                    rocblas_stride stride_alpha,
                    TConstPtr __restrict__ APa,
                    rocblas_stride shifta,
                    rocblas_stride strideA,
                    TConstPtr __restrict__ xa,
                    rocblas_stride shiftx,
                    rocblas_int    incx,
                    rocblas_stride stridex,
                    TScal          beta_device_host,
                    rocblas_stride stride_beta,
                    TPtr __restrict__ ya,
                    rocblas_stride shifty,
                    rocblas_int    incy,
                    rocblas_stride stridey)
{
    rocblas_int num_threads = blockDim.x * blockDim.y * blockDim.z;
    if(DIM_X * DIM_Y != num_threads)
        return; // need to launch exactly the same number of threads as template parameters indicate

    auto alpha = load_scalar(alpha_device_host, blockIdx.y, stride_alpha);
    auto beta  = load_scalar(beta_device_host, blockIdx.y, stride_beta);
    if(!alpha && beta == 1)
        return;

    auto AP = cond_load_ptr_batch(alpha, APa, blockIdx.y, shifta, strideA);
    auto x  = cond_load_ptr_batch(alpha, xa, blockIdx.y, shiftx, stridex);

    auto y = load_ptr_batch(ya, blockIdx.y, shifty, stridey);

    rocblas_spmv_kernel_calc<DIM_X, DIM_Y>(is_upper, n, alpha, AP, x, incx, beta, y, incy);
}

template <typename T, typename U, typename V, typename W>
rocblas_status rocblas_spmv_template(rocblas_handle handle,
                                     rocblas_fill   uplo,
                                     rocblas_int    n,
                                     const V*       alpha,
                                     rocblas_stride stride_alpha,
                                     const U*       A,
                                     rocblas_stride offseta,
                                     rocblas_stride strideA,
                                     const U*       x,
                                     rocblas_stride offsetx,
                                     rocblas_int    incx,
                                     rocblas_stride stridex,
                                     const V*       beta,
                                     rocblas_stride stride_beta,
                                     W*             y,
                                     rocblas_stride offsety,
                                     rocblas_int    incy,
                                     rocblas_stride stridey,
                                     rocblas_int    batch_count)
{
    //quick return
    if(!n || !batch_count)
        return rocblas_status_success;

    hipStream_t rocblas_stream = handle->get_stream();

    // in case of negative inc shift pointer to end of data for negative indexing tid*inc
    auto shiftx = incx < 0 ? offsetx - int64_t(incx) * (n - 1) : offsetx;
    auto shifty = incy < 0 ? offsety - int64_t(incy) * (n - 1) : offsety;

    static constexpr int spmv_DIM_X = 64;
    static constexpr int spmv_DIM_Y = 16;
    rocblas_int          blocks     = (n - 1) / (spmv_DIM_X) + 1;
    dim3                 grid(blocks, batch_count);
    dim3                 threads(spmv_DIM_X, spmv_DIM_Y);

    if(handle->pointer_mode == rocblas_pointer_mode_device)
    {
        hipLaunchKernelGGL((rocblas_spmv_kernel<spmv_DIM_X, spmv_DIM_Y>),
                           grid,
                           threads,
                           0,
                           rocblas_stream,
                           uplo == rocblas_fill_upper,
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

        hipLaunchKernelGGL((rocblas_spmv_kernel<spmv_DIM_X, spmv_DIM_Y>),
                           grid,
                           threads,
                           0,
                           rocblas_stream,
                           uplo == rocblas_fill_upper,
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
                                           rocblas_stride offset_a,
                                           rocblas_stride stride_a,
                                           T              x,
                                           rocblas_stride offset_x,
                                           rocblas_int    inc_x,
                                           rocblas_stride stride_x,
                                           U              y,
                                           rocblas_stride offset_y,
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

// Instantiations below will need to be manually updated to match any change in
// template parameters in the files *spmv*.cpp

// clang-format off

#ifdef INSTANTIATE_SPMV_TEMPLATE
#error INSTANTIATE_SPMV_TEMPLATE already defined
#endif

#define INSTANTIATE_SPMV_TEMPLATE(T_, U_, V_, W_)                     \
template rocblas_status rocblas_spmv_template<T_, U_, V_, W_>         \
                                    (rocblas_handle handle,           \
                                     rocblas_fill   uplo,             \
                                     rocblas_int    n,                \
                                     V_ const*       alpha,           \
                                     rocblas_stride stride_alpha,     \
                                     U_ const*       A,               \
                                     rocblas_stride offseta,          \
                                     rocblas_stride strideA,          \
                                     U_ const*       x,               \
                                     rocblas_stride offsetx,          \
                                     rocblas_int    incx,             \
                                     rocblas_stride stridex,          \
                                     V_ const*       beta,            \
                                     rocblas_stride stride_beta,      \
                                     W_*             y,               \
                                     rocblas_stride offsety,          \
                                     rocblas_int    incy,             \
                                     rocblas_stride stridey,          \
                                     rocblas_int    batch_count);

INSTANTIATE_SPMV_TEMPLATE(float, float, float, float)
INSTANTIATE_SPMV_TEMPLATE(double, double, double, double)
INSTANTIATE_SPMV_TEMPLATE(float, float const*, float, float* const)
INSTANTIATE_SPMV_TEMPLATE(double, double const*, double, double* const)

#undef INSTANTIATE_SPMV_TEMPLATE

#ifdef INSTANTIATE_SPMV_NUMERICS
#error INSTANTIATE_SPMV_NUMERICS already defined
#endif

#define INSTANTIATE_SPMV_NUMERICS(T_, U_)                                   \
template rocblas_status rocblas_spmv_check_numerics<T_, U_>                 \
                                          (const char*    function_name,    \
                                           rocblas_handle handle,           \
                                           rocblas_int    n,                \
                                           T_             A,                \
                                           rocblas_stride    offset_a,         \
                                           rocblas_stride stride_a,         \
                                           T_             x,                \
                                           rocblas_stride    offset_x,         \
                                           rocblas_int    inc_x,            \
                                           rocblas_stride stride_x,         \
                                           U_             y,                \
                                           rocblas_stride    offset_y,         \
                                           rocblas_int    inc_y,            \
                                           rocblas_stride stride_y,         \
                                           rocblas_int    batch_count,      \
                                           const int      check_numerics,   \
                                           bool           is_input);

INSTANTIATE_SPMV_NUMERICS(float const*, float*)
INSTANTIATE_SPMV_NUMERICS(double const*, double*)
INSTANTIATE_SPMV_NUMERICS(float const* const*, float* const*)
INSTANTIATE_SPMV_NUMERICS(double const* const*, double* const*)

#undef INSTANTIATE_SPMV_NUMERICS

// clang-format on
