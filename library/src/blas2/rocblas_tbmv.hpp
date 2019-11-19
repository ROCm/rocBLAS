/* ************************************************************************
 * Copyright 2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#ifndef __ROCBLAS_TBMV_HPP__
#define __ROCBLAS_TBMV_HPP__
#include "handle.h"
#include "rocblas.h"

template <typename T, typename V>
rocblas_status rocblas_tbmv_template(rocblas_handle    handle,
                                     rocblas_fill      uplo,
                                     rocblas_operation transA,
                                     rocblas_diagonal  diag,
                                     rocblas_int       m,
                                     rocblas_int       k,
                                     const V*          A,
                                     rocblas_int       offseta,
                                     rocblas_int       lda,
                                     rocblas_stride    strideA,
                                     const V*          x,
                                     rocblas_int       offsetx,
                                     rocblas_int       incx,
                                     rocblas_stride    stridex,
                                     rocblas_int       batch_count)
{
    // //quick return
    // if(!m || !batch_count)
    //     return rocblas_status_success;

    // hipStream_t rocblas_stream = handle->rocblas_stream;

    // // in case of negative inc shift pointer to end of data for negative indexing tid*inc
    // auto shiftx
    //     = incx < 0 ? offsetx - ptrdiff_t(incx) * (transA == rocblas_operation_none ? n - 1 : m - 1)
    //                : offsetx;
    // auto shifty
    //     = incy < 0 ? offsety - ptrdiff_t(incy) * (transA == rocblas_operation_none ? m - 1 : n - 1)
    //                : offsety;

    // if(transA == rocblas_operation_none)
    // {
    //     // TBMVN_DIM_Y must be at least 4, 8 * 8 is very slow only 40Gflop/s
    //     static constexpr int TBMVN_DIM_X = 64;
    //     static constexpr int TBMVN_DIM_Y = 16;
    //     rocblas_int          blocks      = (m - 1) / (TBMVN_DIM_X * 4) + 1;
    //     if(std::is_same<T, rocblas_double_complex>{})
    //         blocks = (m - 1) / (TBMVN_DIM_X) + 1;
    //     dim3 tbmvn_grid(blocks, batch_count);
    //     dim3 tbmvn_threads(TBMVN_DIM_X, TBMVN_DIM_Y);

    //     if(handle->pointer_mode == rocblas_pointer_mode_device)
    //     {
    //         hipLaunchKernelGGL((tbmvn_kernel<TBMVN_DIM_X, TBMVN_DIM_Y, T>),
    //                            tbmvn_grid,
    //                            tbmvn_threads,
    //                            0,
    //                            rocblas_stream,
    //                            m,
    //                            n,
    //                            alpha,
    //                            stride_alpha,
    //                            A,
    //                            offseta,
    //                            lda,
    //                            strideA,
    //                            x,
    //                            shiftx,
    //                            incx,
    //                            stridex,
    //                            beta,
    //                            stride_beta,
    //                            y,
    //                            shifty,
    //                            incy,
    //                            stridey);
    //     }
    //     else
    //     {
    //         if(!*alpha && *beta == 1)
    //             return rocblas_status_success;

    //         hipLaunchKernelGGL((tbmvn_kernel<TBMVN_DIM_X, TBMVN_DIM_Y, T>),
    //                            tbmvn_grid,
    //                            tbmvn_threads,
    //                            0,
    //                            rocblas_stream,
    //                            m,
    //                            n,
    //                            *alpha,
    //                            stride_alpha,
    //                            A,
    //                            offseta,
    //                            lda,
    //                            strideA,
    //                            x,
    //                            shiftx,
    //                            incx,
    //                            stridex,
    //                            *beta,
    //                            stride_beta,
    //                            y,
    //                            shifty,
    //                            incy,
    //                            stridey);
    //     }
    // }
    // else if(transA == rocblas_operation_transpose)
    // {
    //     // transpose
    //     // number of columns on the y-dim of the grid
    //     static constexpr int NB = 256;
    //     dim3                 tbmvt_grid(n, batch_count);
    //     dim3                 tbmvt_threads(NB);

    //     if(handle->pointer_mode == rocblas_pointer_mode_device)
    //     {
    //         hipLaunchKernelGGL((tbmvt_kernel<NB, T>),
    //                            tbmvt_grid,
    //                            tbmvt_threads,
    //                            0,
    //                            rocblas_stream,
    //                            m,
    //                            n,
    //                            alpha,
    //                            stride_alpha,
    //                            A,
    //                            offseta,
    //                            lda,
    //                            strideA,
    //                            x,
    //                            shiftx,
    //                            incx,
    //                            stridex,
    //                            beta,
    //                            stride_beta,
    //                            y,
    //                            shifty,
    //                            incy,
    //                            stridey);
    //     }
    //     else
    //     {
    //         if(!*alpha && *beta == 1)
    //             return rocblas_status_success;

    //         hipLaunchKernelGGL((tbmvt_kernel<NB, T>),
    //                            tbmvt_grid,
    //                            tbmvt_threads,
    //                            0,
    //                            rocblas_stream,
    //                            m,
    //                            n,
    //                            *alpha,
    //                            stride_alpha,
    //                            A,
    //                            offseta,
    //                            lda,
    //                            strideA,
    //                            x,
    //                            shiftx,
    //                            incx,
    //                            stridex,
    //                            *beta,
    //                            stride_beta,
    //                            y,
    //                            shifty,
    //                            incy,
    //                            stridey);
    //     }
    // }
    // else // conjugate transpose
    // {
    //     // conjugate transpose
    //     // number of columns on the y-dim of the grid
    //     static constexpr int NB = 256;
    //     dim3                 tbmvc_grid(n, batch_count);
    //     dim3                 tbmvc_threads(NB);

    //     if(handle->pointer_mode == rocblas_pointer_mode_device)
    //     {
    //         hipLaunchKernelGGL((tbmvc_kernel<NB, T>),
    //                            tbmvc_grid,
    //                            tbmvc_threads,
    //                            0,
    //                            rocblas_stream,
    //                            m,
    //                            n,
    //                            alpha,
    //                            stride_alpha,
    //                            A,
    //                            offseta,
    //                            lda,
    //                            strideA,
    //                            x,
    //                            shiftx,
    //                            incx,
    //                            stridex,
    //                            beta,
    //                            stride_beta,
    //                            y,
    //                            shifty,
    //                            incy,
    //                            stridey);
    //     }
    //     else
    //     {
    //         if(!*alpha && *beta == 1)
    //             return rocblas_status_success;

    //         hipLaunchKernelGGL((tbmvc_kernel<NB, T>),
    //                            tbmvc_grid,
    //                            tbmvc_threads,
    //                            0,
    //                            rocblas_stream,
    //                            m,
    //                            n,
    //                            *alpha,
    //                            stride_alpha,
    //                            A,
    //                            offseta,
    //                            lda,
    //                            strideA,
    //                            x,
    //                            shiftx,
    //                            incx,
    //                            stridex,
    //                            *beta,
    //                            stride_beta,
    //                            y,
    //                            shifty,
    //                            incy,
    //                            stridey);
    //     }
    // }
    return rocblas_status_success;
}

#endif
