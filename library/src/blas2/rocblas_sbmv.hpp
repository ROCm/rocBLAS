/* ************************************************************************
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#pragma once
#include "handle.h"
#include "rocblas.h"

template <typename T, typename U, typename V, typename W>
__global__ void rocblas_sbmv_kernel(rocblas_fill   uplo,
                                    rocblas_int    n,
                                    rocblas_int    k,
                                    V              alpha_device_host,
                                    rocblas_stride stride_alpha,
                                    const U __restrict__ Aa,
                                    rocblas_int    shiftA,
                                    rocblas_int    lda,
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

        int dl = tx - k;
        dl     = dl < 0 ? 0 : dl;
        int dr = tx + k;
        dr     = dr > n - 1 ? n - 1 : dr;

        int dd = (uplo == rocblas_fill_upper) ? k : 0;

        int from = (uplo == rocblas_fill_upper) ? tx : dl;
        int to   = (uplo == rocblas_fill_upper) ? dr : tx;
        for(int i = from; i <= to; i++)
        {
            int c = i;
            int r = tx;
            r     = r - c + dd;

            dotp += x[i * incx] * A[r + c * lda];
        }

        from = (uplo == rocblas_fill_upper) ? dl : tx + 1;
        to   = (uplo == rocblas_fill_upper) ? tx - 1 : dr;
        for(int i = from; i <= to; i++)
        {
            // transpose A fetch for opposite side of triangle
            int c = tx;
            int r = i;
            r     = r - c + dd;

            dotp += x[i * incx] * A[r + c * lda];
        }

        y[tx * incy] = alpha * dotp + beta * y[tx * incy];
    }
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

    if(n < 0 || k < 0 || lda < n || lda < 1 || !incx || !incy || batch_count < 0)
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

    static constexpr int sbmv_DIM_Y = 512;
    rocblas_int          blocks     = (n - 1) / (sbmv_DIM_Y) + 1;
    dim3                 grid(blocks, batch_count);
    dim3                 threads(sbmv_DIM_Y);

    if(handle->pointer_mode == rocblas_pointer_mode_device)
    {
        hipLaunchKernelGGL(rocblas_sbmv_kernel<T>,
                           grid,
                           threads,
                           0,
                           rocblas_stream,
                           uplo,
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
        // quick return only for non-batched
        if(batch_count == 1 && !*alpha && *beta == 1)
            return rocblas_status_success;

        hipLaunchKernelGGL(rocblas_sbmv_kernel<T>,
                           grid,
                           threads,
                           0,
                           rocblas_stream,
                           uplo,
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

    return rocblas_status_success;
}
