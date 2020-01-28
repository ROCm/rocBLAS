/* ************************************************************************
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#pragma once

#include "handle.h"
#include "rocblas.h"

template <typename T, typename U, typename V, typename W>
__global__ void rocblas_spmv_kernel(rocblas_fill   uplo,
                                    rocblas_int    n,
                                    V              alpha_device_host,
                                    rocblas_stride stride_alpha,
                                    const U __restrict__ Aa,
                                    rocblas_int    shiftA,
                                    rocblas_stride strideA,
                                    const U __restrict__ xa,
                                    ptrdiff_t      shiftx,
                                    rocblas_int    incx,
                                    rocblas_stride stridex,
                                    V              beta_device_host,
                                    rocblas_stride stride_beta,
                                    W              ya,
                                    ptrdiff_t      shifty,
                                    rocblas_int    incy,
                                    rocblas_stride stridey)
{
    ptrdiff_t tx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(tx < n)
    {
        auto alpha              = load_scalar(alpha_device_host, hipBlockIdx_y, stride_alpha);
        auto beta               = load_scalar(beta_device_host, hipBlockIdx_y, stride_beta);
        const T* __restrict__ A = load_ptr_batch(Aa, hipBlockIdx_y, shiftA, strideA);
        const T* __restrict__ x = load_ptr_batch(xa, hipBlockIdx_y, shiftx, stridex);
        T* y                    = load_ptr_batch(ya, hipBlockIdx_y, shifty, stridey);

        T dotp = 0;

        int from = uplo == rocblas_fill_upper ? tx : 0;
        int to   = uplo == rocblas_fill_upper ? n - 1 : tx - 1;
        for(int j = from; j <= to; j++)
        {
            int idx = (uplo == rocblas_fill_lower) ? tx + (2 * n - (j + 1)) * (j) / 2
                                                   : tx + (j + 1) * (j) / 2;
            dotp += x[j * incx] * A[idx];
        }

        from = uplo == rocblas_fill_upper ? 0 : tx;
        to   = uplo == rocblas_fill_upper ? tx - 1 : n - 1;
        for(int j = from; j <= to; j++)
        {
            // transpose A fetch for opposite side of triangle
            int idx = (uplo == rocblas_fill_lower) ? j + (2 * n - (tx + 1)) * (tx) / 2
                                                   : j + (tx + 1) * (tx) / 2;
            dotp += x[j * incx] * A[idx];
        }

        y[tx * incy] = dotp * alpha + beta * y[tx * incy];
    }
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

    hipStream_t rocblas_stream = handle->rocblas_stream;

    // in case of negative inc shift pointer to end of data for negative indexing tid*inc
    auto shiftx = incx < 0 ? offsetx - ptrdiff_t(incx) * (n - 1) : offsetx;
    auto shifty = incy < 0 ? offsety - ptrdiff_t(incy) * (n - 1) : offsety;

    static constexpr int spmv_DIM_Y = 512;
    rocblas_int          blocks     = (n - 1) / (spmv_DIM_Y) + 1;
    dim3                 grid(blocks, batch_count);
    dim3                 threads(spmv_DIM_Y);

    if(handle->pointer_mode == rocblas_pointer_mode_device)
    {
        hipLaunchKernelGGL(rocblas_spmv_kernel<T>,
                           grid,
                           threads,
                           0,
                           rocblas_stream,
                           uplo,
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

        hipLaunchKernelGGL(rocblas_spmv_kernel<T>,
                           grid,
                           threads,
                           0,
                           rocblas_stream,
                           uplo,
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
