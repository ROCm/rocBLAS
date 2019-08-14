/* ************************************************************************
 * Copyright 2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#ifndef __ROCBLAS_GEMV_HPP__
#define __ROCBLAS_GEMV_HPP__
#include "gemv_device.hpp"
#include "handle.h"
#include "rocblas.h"

template <typename T>
rocblas_status rocblas_gemv_template(rocblas_handle    handle,
                                     rocblas_operation transA,
                                     rocblas_int       m,
                                     rocblas_int       n,
                                     const T*          alpha,
                                     const T*          A,
                                     rocblas_int       lda,
                                     const T*          x,
                                     rocblas_int       incx,
                                     const T*          beta,
                                     T*                y,
                                     rocblas_int       incy)
{
    //quick return
    if(!m || !n)
        return rocblas_status_success;

    hipStream_t rocblas_stream = handle->rocblas_stream;

    if(transA == rocblas_operation_none)
    {
        // GEMVN_DIM_Y must be at least 4, 8 * 8 is very slow only 40Gflop/s
        static constexpr int GEMVN_DIM_X = 64;
        static constexpr int GEMVN_DIM_Y = 16;
        rocblas_int          blocks      = (m - 1) / (GEMVN_DIM_X * 4) + 1;
        if(std::is_same<T, rocblas_double_complex>{})
            blocks = (m - 1) / (GEMVN_DIM_X) + 1;
        dim3 gemvn_grid(blocks, 1);
        dim3 gemvn_threads(GEMVN_DIM_X, GEMVN_DIM_Y);

        if(handle->pointer_mode == rocblas_pointer_mode_device)
        {
            hipLaunchKernelGGL((gemvn_kernel_strided<GEMVN_DIM_X, GEMVN_DIM_Y>),
                               gemvn_grid,
                               gemvn_threads,
                               0,
                               rocblas_stream,
                               m,
                               n,
                               alpha,
                               A,
                               lda,
                               0, // strideA = 0
                               x,
                               incx,
                               0, // stridex = 0
                               beta,
                               y,
                               incy,
                               0); // stridey = 0
        }
        else
        {
            if(!*alpha && *beta == 1)
                return rocblas_status_success;

            hipLaunchKernelGGL((gemvn_kernel_strided<GEMVN_DIM_X, GEMVN_DIM_Y>),
                               gemvn_grid,
                               gemvn_threads,
                               0,
                               rocblas_stream,
                               m,
                               n,
                               *alpha,
                               A,
                               lda,
                               0, // strideA = 0
                               x,
                               incx,
                               0, // stridex = 0
                               *beta,
                               y,
                               incy,
                               0); // stridey = 0
        }
    }
    else if(transA == rocblas_operation_transpose)
    {
        // transpose
        // number of columns on the y-dim of the grid
        static constexpr int NB = 256;
        dim3                 gemvt_grid(n, 1);
        dim3                 gemvt_threads(NB);

        if(handle->pointer_mode == rocblas_pointer_mode_device)
        {
            hipLaunchKernelGGL(gemvt_kernel_strided<NB>,
                               gemvt_grid,
                               gemvt_threads,
                               0,
                               rocblas_stream,
                               m,
                               n,
                               alpha,
                               A,
                               lda,
                               0, // strideA = 0
                               x,
                               incx,
                               0, // stridex = 0
                               beta,
                               y,
                               incy,
                               0); // stridey = 0
        }
        else
        {
            if(!*alpha && *beta == 1)
                return rocblas_status_success;

            hipLaunchKernelGGL(gemvt_kernel_strided<NB>,
                               gemvt_grid,
                               gemvt_threads,
                               0,
                               rocblas_stream,
                               m,
                               n,
                               *alpha,
                               A,
                               lda,
                               0, // strideA = 0
                               x,
                               incx,
                               0, // stridex = 0
                               *beta,
                               y,
                               incy,
                               0); // stridey = 0
        }
    }
    else // conjugate transpose
    {
        // conjugate transpose
        // number of columns on the y-dim of the grid
        static constexpr int NB = 256;
        dim3                 gemvc_grid(n, 1);
        dim3                 gemvc_threads(NB);

        if(handle->pointer_mode == rocblas_pointer_mode_device)
        {
            hipLaunchKernelGGL(gemvc_kernel_strided<NB>,
                               gemvc_grid,
                               gemvc_threads,
                               0,
                               rocblas_stream,
                               m,
                               n,
                               alpha,
                               A,
                               lda,
                               0, // strideA = 0
                               x,
                               incx,
                               0, // stridex = 0
                               beta,
                               y,
                               incy,
                               0); // stridey = 0
        }
        else
        {
            if(!*alpha && *beta == 1)
                return rocblas_status_success;

            hipLaunchKernelGGL(gemvc_kernel_strided<NB>,
                               gemvc_grid,
                               gemvc_threads,
                               0,
                               rocblas_stream,
                               m,
                               n,
                               *alpha,
                               A,
                               lda,
                               0, // strideA = 0
                               x,
                               incx,
                               0, // stridex = 0
                               *beta,
                               y,
                               incy,
                               0); // stridey = 0
        }
    }
    return rocblas_status_success;
}

template <typename T>
rocblas_status rocblas_gemv_batched_template(rocblas_handle    handle,
                                             rocblas_operation transA,
                                             rocblas_int       m,
                                             rocblas_int       n,
                                             const T*          alpha,
                                             const T* const    A[],
                                             rocblas_int       lda,
                                             const T* const    x[],
                                             rocblas_int       incx,
                                             const T*          beta,
                                             T* const          y[],
                                             rocblas_int       incy,
                                             rocblas_int       batch_count)
{
    // Quick return if possible. Not Argument error
    if(!m || !n || !batch_count)
        return rocblas_status_success;

    hipStream_t rocblas_stream = handle->rocblas_stream;

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
            hipLaunchKernelGGL((gemvn_kernel_batched<GEMVN_DIM_X, GEMVN_DIM_Y>),
                               gemvn_grid,
                               gemvn_threads,
                               0,
                               rocblas_stream,
                               m,
                               n,
                               alpha,
                               A,
                               lda,
                               x,
                               incx,
                               beta,
                               y,
                               incy);
        }
        else
        {
            if(!*alpha && *beta == 1)
                return rocblas_status_success;

            hipLaunchKernelGGL((gemvn_kernel_batched<GEMVN_DIM_X, GEMVN_DIM_Y>),
                               gemvn_grid,
                               gemvn_threads,
                               0,
                               rocblas_stream,
                               m,
                               n,
                               *alpha,
                               A,
                               lda,
                               x,
                               incx,
                               *beta,
                               y,
                               incy);
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
            hipLaunchKernelGGL(gemvt_kernel_batched<NB>,
                               gemvt_grid,
                               gemvt_threads,
                               0,
                               rocblas_stream,
                               m,
                               n,
                               alpha,
                               A,
                               lda,
                               x,
                               incx,
                               beta,
                               y,
                               incy);
        }
        else
        {
            if(!*alpha && *beta == 1)
                return rocblas_status_success;

            hipLaunchKernelGGL(gemvt_kernel_batched<NB>,
                               gemvt_grid,
                               gemvt_threads,
                               0,
                               rocblas_stream,
                               m,
                               n,
                               *alpha,
                               A,
                               lda,
                               x,
                               incx,
                               *beta,
                               y,
                               incy);
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
            hipLaunchKernelGGL(gemvc_kernel_batched<NB>,
                               gemvc_grid,
                               gemvc_threads,
                               0,
                               rocblas_stream,
                               m,
                               n,
                               alpha,
                               A,
                               lda,
                               x,
                               incx,
                               beta,
                               y,
                               incy);
        }
        else
        {
            if(!*alpha && *beta == 1)
                return rocblas_status_success;

            hipLaunchKernelGGL(gemvc_kernel_batched<NB>,
                               gemvc_grid,
                               gemvc_threads,
                               0,
                               rocblas_stream,
                               m,
                               n,
                               *alpha,
                               A,
                               lda,
                               x,
                               incx,
                               *beta,
                               y,
                               incy);
        }
    }

    return rocblas_status_success;
}

template <typename T>
rocblas_status rocblas_gemv_strided_batched_template(rocblas_handle    handle,
                                                     rocblas_operation transA,
                                                     rocblas_int       m,
                                                     rocblas_int       n,
                                                     const T*          alpha,
                                                     const T*          A,
                                                     rocblas_int       lda,
                                                     rocblas_int       strideA,
                                                     const T*          x,
                                                     rocblas_int       incx,
                                                     rocblas_int       stridex,
                                                     const T*          beta,
                                                     T*                y,
                                                     rocblas_int       incy,
                                                     rocblas_int       stridey,
                                                     rocblas_int       batch_count)
{
    // Quick return if possible. Not Argument error
    if(!m || !n || !batch_count)
        return rocblas_status_success;

    hipStream_t rocblas_stream = handle->rocblas_stream;

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
            hipLaunchKernelGGL((gemvn_kernel_strided<GEMVN_DIM_X, GEMVN_DIM_Y>),
                               gemvn_grid,
                               gemvn_threads,
                               0,
                               rocblas_stream,
                               m,
                               n,
                               alpha,
                               A,
                               lda,
                               strideA,
                               x,
                               incx,
                               stridex,
                               beta,
                               y,
                               incy,
                               stridey);
        }
        else
        {
            if(!*alpha && *beta == 1)
                return rocblas_status_success;

            hipLaunchKernelGGL((gemvn_kernel_strided<GEMVN_DIM_X, GEMVN_DIM_Y>),
                               gemvn_grid,
                               gemvn_threads,
                               0,
                               rocblas_stream,
                               m,
                               n,
                               *alpha,
                               A,
                               lda,
                               strideA,
                               x,
                               incx,
                               stridex,
                               *beta,
                               y,
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
            hipLaunchKernelGGL(gemvt_kernel_strided<NB>,
                               gemvt_grid,
                               gemvt_threads,
                               0,
                               rocblas_stream,
                               m,
                               n,
                               alpha,
                               A,
                               lda,
                               strideA,
                               x,
                               incx,
                               stridex,
                               beta,
                               y,
                               incy,
                               stridey);
        }
        else
        {
            if(!*alpha && *beta == 1)
                return rocblas_status_success;

            hipLaunchKernelGGL(gemvt_kernel_strided<NB>,
                               gemvt_grid,
                               gemvt_threads,
                               0,
                               rocblas_stream,
                               m,
                               n,
                               *alpha,
                               A,
                               lda,
                               strideA,
                               x,
                               incx,
                               stridex,
                               *beta,
                               y,
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
            hipLaunchKernelGGL(gemvc_kernel_strided<NB>,
                               gemvc_grid,
                               gemvc_threads,
                               0,
                               rocblas_stream,
                               m,
                               n,
                               alpha,
                               A,
                               lda,
                               strideA,
                               x,
                               incx,
                               stridex,
                               beta,
                               y,
                               incy,
                               stridey);
        }
        else
        {
            if(!*alpha && *beta == 1)
                return rocblas_status_success;

            hipLaunchKernelGGL(gemvc_kernel_strided<NB>,
                               gemvc_grid,
                               gemvc_threads,
                               0,
                               rocblas_stream,
                               m,
                               n,
                               *alpha,
                               A,
                               lda,
                               strideA,
                               x,
                               incx,
                               stridex,
                               *beta,
                               y,
                               incy,
                               stridey);
        }
    }
    return rocblas_status_success;
}

#endif
