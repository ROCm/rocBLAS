/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "definitions.h"
#include "gemv_device.hpp"
#include "handle.h"
#include "logging.h"
#include "rocblas.h"
#include "status.h"
#include "utility.h"

namespace
{
    template <typename>
    static constexpr char rocblas_gemv_name[] = "unknown";
    template <>
    static constexpr char rocblas_gemv_name<float>[] = "rocblas_sgemv_strided_batched";
    template <>
    static constexpr char rocblas_gemv_name<double>[] = "rocblas_dgemv_strided_batched";

    template <typename T>
    rocblas_status rocblas_gemv_strided_batched(rocblas_handle    handle,
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
        if(!handle)
            return rocblas_status_invalid_handle;
        if(!alpha || !beta)
            return rocblas_status_invalid_pointer;

        auto layer_mode = handle->layer_mode;
        if(layer_mode
           & (rocblas_layer_mode_log_trace | rocblas_layer_mode_log_bench
              | rocblas_layer_mode_log_profile))
        {
            auto transA_letter = rocblas_transpose_letter(transA);

            if(handle->pointer_mode == rocblas_pointer_mode_host)
            {
                if(layer_mode & rocblas_layer_mode_log_trace)
                    log_trace(handle,
                              rocblas_gemv_name<T>,
                              transA,
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
                              stridey,
                              batch_count);

                if(layer_mode & rocblas_layer_mode_log_bench)
                    log_bench(handle,
                              "./rocblas-bench -f gemv_strided_batched -r",
                              rocblas_precision_string<T>,
                              "--transposeA",
                              transA_letter,
                              "-m",
                              m,
                              "-n",
                              n,
                              "--alpha",
                              *alpha,
                              "--lda",
                              lda,
                              "--strideA",
                              strideA,
                              "--incx",
                              incx,
                              "--stridex",
                              stridex,
                              "--beta",
                              *beta,
                              "--incy",
                              incy,
                              "--stridey",
                              stridey,
                              "--batch_count",
                              batch_count);
            }
            else
            {
                if(layer_mode & rocblas_layer_mode_log_trace)
                    log_trace(handle,
                              rocblas_gemv_name<T>,
                              transA,
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
                              stridey,
                              batch_count);
            }

            if(layer_mode & rocblas_layer_mode_log_profile)
                log_profile(handle,
                            rocblas_gemv_name<T>,
                            "transA",
                            transA_letter,
                            "M",
                            m,
                            "N",
                            n,
                            "lda",
                            lda,
                            "strideA",
                            strideA,
                            "incx",
                            incx,
                            "stridex",
                            stridex,
                            "incy",
                            incy,
                            "stridey",
                            stridey,
                            "batch_count",
                            batch_count);
        }

        if(!A || !x || !y)
            return rocblas_status_invalid_pointer;
        if(m < 0 || n < 0 || lda < m || lda < 1 || !incx || !incy)
            return rocblas_status_invalid_size;
        // Quick return if possible. Not Argument error
        if(!m || !n)
            return rocblas_status_success;

        hipStream_t rocblas_stream = handle->rocblas_stream;

        if(transA == rocblas_operation_none)
        {
            // GEMVN_DIM_Y must be at least 4, 8 * 8 is very slow only 40Gflop/s
            static constexpr int GEMVN_DIM_X = 64;
            static constexpr int GEMVN_DIM_Y = 16;
            rocblas_int          blocks      = (m - 1) / (GEMVN_DIM_X * 4) + 1;

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
        else
        {
            // transpose
            // number of columns on the y-dim of the grid, using gemvc because gemvt(transpose) is a
            // instance of gemvc (conjugate)
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

} // namespace

/*
* ===========================================================================
*    C wrapper
* ===========================================================================
*/

extern "C" {

rocblas_status rocblas_sgemv_strided_batched(rocblas_handle    handle,
                                             rocblas_operation transA,
                                             rocblas_int       m,
                                             rocblas_int       n,
                                             const float*      alpha,
                                             const float*      A,
                                             rocblas_int       lda,
                                             rocblas_int       strideA,
                                             const float*      x,
                                             rocblas_int       incx,
                                             rocblas_int       stridex,
                                             const float*      beta,
                                             float*            y,
                                             rocblas_int       incy,
                                             rocblas_int       stridey,
                                             rocblas_int       batch_count)
{
    return rocblas_gemv_strided_batched(handle,
                                        transA,
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
                                        stridey,
                                        batch_count);
}

rocblas_status rocblas_dgemv_strided_batched(rocblas_handle    handle,
                                             rocblas_operation transA,
                                             rocblas_int       m,
                                             rocblas_int       n,
                                             const double*     alpha,
                                             const double*     A,
                                             rocblas_int       lda,
                                             rocblas_int       strideA,
                                             const double*     x,
                                             rocblas_int       incx,
                                             rocblas_int       stridex,
                                             const double*     beta,
                                             double*           y,
                                             rocblas_int       incy,
                                             rocblas_int       stridey,
                                             rocblas_int       batch_count)
{
    return rocblas_gemv_strided_batched(handle,
                                        transA,
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
                                        stridey,
                                        batch_count);
}

} // extern "C"
