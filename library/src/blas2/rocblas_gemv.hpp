/* ************************************************************************
 * Copyright 2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#ifndef __ROCBLAS_GEMV_HPP__
#define __ROCBLAS_GEMV_HPP__
#include "gemv_device.hpp"
#include "handle.h"
#include "rocblas.h"

template <typename T, typename U, typename V, typename W>
rocblas_status rocblas_gemv_template(rocblas_handle    handle,
                                     rocblas_operation transA,
                                     rocblas_int       m,
                                     rocblas_int       n,
                                     const U*          alpha,
                                     const V*          A,
                                     rocblas_int       offseta,
                                     rocblas_int       lda,
                                     rocblas_stride    strideA,
                                     const V*          x,
                                     rocblas_int       offsetx,
                                     rocblas_int       incx,
                                     rocblas_stride    stridex,
                                     const U*          beta,
                                     W*                y,
                                     rocblas_int       offsety,
                                     rocblas_int       incy,
                                     rocblas_stride    stridey,
                                     rocblas_int       batch_count)
{
    //quick return
    if(!m || !n || !batch_count)
        return rocblas_status_success;

    hipStream_t rocblas_stream = handle->rocblas_stream;

    // in case of negative inc shift pointer to end of data for negative indexing tid*inc
    auto shiftx = incx < 0
                      ? offsetx - ptrdiff_t(incx) * ((transA == rocblas_operation_none ? n : m) - 1)
                      : offsetx;
    auto shifty = incy < 0
                      ? offsety - ptrdiff_t(incy) * ((transA == rocblas_operation_none ? m : n) - 1)
                      : offsety;

    if(transA == rocblas_operation_none)
    {
        // GEMVN_DIM_Y must be at least 4, 8 * 8 is very slow only 40Gflop/s
        static constexpr int GEMVN_DIM_X = 64;
        static constexpr int GEMVN_DIM_Y = 16;
        rocblas_int          blocks      = (m - 1) / (GEMVN_DIM_X * 4) + 1;
        if(std::is_same<T, rocblas_double_complex>{})
            blocks = (m - 1) / (GEMVN_DIM_X) + 1;
        dim3 gemvn_grid(blocks, batch_count);
        dim3 gemvn_threads(GEMVN_DIM_X, GEMVN_DIM_Y);

        if(handle->pointer_mode == rocblas_pointer_mode_device)
        {
            hipLaunchKernelGGL((gemvn_kernel<GEMVN_DIM_X, GEMVN_DIM_Y, T>),
                               gemvn_grid,
                               gemvn_threads,
                               0,
                               rocblas_stream,
                               m,
                               n,
                               alpha,
                               0,
                               A,
                               offseta,
                               lda,
                               strideA,
                               x,
                               shiftx,
                               incx,
                               stridex,
                               beta,
                               0,
                               y,
                               shifty,
                               incy,
                               stridey);
        }
        else
        {
            if(!*alpha && *beta == 1)
                return rocblas_status_success;

            hipLaunchKernelGGL((gemvn_kernel<GEMVN_DIM_X, GEMVN_DIM_Y, T>),
                               gemvn_grid,
                               gemvn_threads,
                               0,
                               rocblas_stream,
                               m,
                               n,
                               *alpha,
                               0,
                               A,
                               offseta,
                               lda,
                               strideA,
                               x,
                               shiftx,
                               incx,
                               stridex,
                               *beta,
                               0,
                               y,
                               shifty,
                               incy,
                               stridey);
        }
    }
    else if(transA == rocblas_operation_transpose)
    {
        // transpose
        // number of columns on the y-dim of the grid
        static constexpr int NB = 256;
        dim3                 gemvt_grid(n, batch_count);
        dim3                 gemvt_threads(NB);

        if(handle->pointer_mode == rocblas_pointer_mode_device)
        {
            hipLaunchKernelGGL((gemvt_kernel<NB, T>),
                               gemvt_grid,
                               gemvt_threads,
                               0,
                               rocblas_stream,
                               m,
                               n,
                               alpha,
                               0,
                               A,
                               offseta,
                               lda,
                               strideA,
                               x,
                               shiftx,
                               incx,
                               stridex,
                               beta,
                               0,
                               y,
                               shifty,
                               incy,
                               stridey);
        }
        else
        {
            if(!*alpha && *beta == 1)
                return rocblas_status_success;

            hipLaunchKernelGGL((gemvt_kernel<NB, T>),
                               gemvt_grid,
                               gemvt_threads,
                               0,
                               rocblas_stream,
                               m,
                               n,
                               *alpha,
                               0,
                               A,
                               offseta,
                               lda,
                               strideA,
                               x,
                               shiftx,
                               incx,
                               stridex,
                               *beta,
                               0,
                               y,
                               shifty,
                               incy,
                               stridey);
        }
    }
    else // conjugate transpose
    {
        // conjugate transpose
        // number of columns on the y-dim of the grid
        static constexpr int NB = 256;
        dim3                 gemvc_grid(n, batch_count);
        dim3                 gemvc_threads(NB);

        if(handle->pointer_mode == rocblas_pointer_mode_device)
        {
            hipLaunchKernelGGL((gemvc_kernel<NB, T>),
                               gemvc_grid,
                               gemvc_threads,
                               0,
                               rocblas_stream,
                               m,
                               n,
                               alpha,
                               0,
                               A,
                               offseta,
                               lda,
                               strideA,
                               x,
                               shiftx,
                               incx,
                               stridex,
                               beta,
                               0,
                               y,
                               shifty,
                               incy,
                               stridey);
        }
        else
        {
            if(!*alpha && *beta == 1)
                return rocblas_status_success;

            hipLaunchKernelGGL((gemvc_kernel<NB, T>),
                               gemvc_grid,
                               gemvc_threads,
                               0,
                               rocblas_stream,
                               m,
                               n,
                               *alpha,
                               0,
                               A,
                               offseta,
                               lda,
                               strideA,
                               x,
                               shiftx,
                               incx,
                               stridex,
                               *beta,
                               0,
                               y,
                               shifty,
                               incy,
                               stridey);
        }
    }
    return rocblas_status_success;
}

#endif
