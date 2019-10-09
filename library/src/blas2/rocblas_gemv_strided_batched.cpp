/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "handle.h"
#include "logging.h"
#include "rocblas.h"
#include "rocblas_gemv.hpp"
#include "utility.h"

namespace
{
    template <typename>
    constexpr char rocblas_gemv_name[] = "unknown";
    template <>
    constexpr char rocblas_gemv_name<float>[] = "rocblas_sgemv_strided_batched";
    template <>
    constexpr char rocblas_gemv_name<double>[] = "rocblas_dgemv_strided_batched";
    template <>
    constexpr char rocblas_gemv_name<rocblas_float_complex>[] = "rocblas_cgemv_strided_batched";
    template <>
    constexpr char rocblas_gemv_name<rocblas_double_complex>[] = "rocblas_zgemv_strided_batched";

    template <typename T>
    rocblas_status rocblas_gemv_strided_batched_impl(rocblas_handle    handle,
                                                     rocblas_operation transA,
                                                     rocblas_int       m,
                                                     rocblas_int       n,
                                                     const T*          alpha,
                                                     const T*          A,
                                                     rocblas_int       lda,
                                                     rocblas_stride    strideA,
                                                     const T*          x,
                                                     rocblas_int       incx,
                                                     rocblas_stride    stridex,
                                                     const T*          beta,
                                                     T*                y,
                                                     rocblas_int       incy,
                                                     rocblas_stride    stridey,
                                                     rocblas_int       batch_count)
    {
        if(!handle)
            return rocblas_status_invalid_handle;
        RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);

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
                              log_trace_scalar_value(alpha),
                              A,
                              lda,
                              strideA,
                              x,
                              incx,
                              stridex,
                              log_trace_scalar_value(beta),
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
                              LOG_BENCH_SCALAR_VALUE(alpha),
                              "--lda",
                              lda,
                              "--stride_a",
                              strideA,
                              "--incx",
                              incx,
                              "--stride_x",
                              stridex,
                              LOG_BENCH_SCALAR_VALUE(beta),
                              "--incy",
                              incy,
                              "--stride_y",
                              stridey,
                              "--batch",
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
                            "stride_a",
                            strideA,
                            "incx",
                            incx,
                            "stride_x",
                            stridex,
                            "incy",
                            incy,
                            "stride_y",
                            stridey,
                            "batch_count",
                            batch_count);
        }

        if(m < 0 || n < 0 || lda < m || lda < 1 || !incx || !incy || batch_count < 0)
            return rocblas_status_invalid_size;
        if(!m || !n || !batch_count)
            return rocblas_status_success;
        if(!A || !x || !y || !alpha || !beta)
            return rocblas_status_invalid_pointer;

        return rocblas_gemv_template<T>(handle,
                                        transA,
                                        m,
                                        n,
                                        alpha,
                                        A,
                                        0,
                                        lda,
                                        strideA,
                                        x,
                                        0,
                                        incx,
                                        stridex,
                                        beta,
                                        y,
                                        0,
                                        incy,
                                        stridey,
                                        batch_count);
    }
} //namespace

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
                                             rocblas_stride    strideA,
                                             const float*      x,
                                             rocblas_int       incx,
                                             rocblas_stride    stridex,
                                             const float*      beta,
                                             float*            y,
                                             rocblas_int       incy,
                                             rocblas_stride    stridey,
                                             rocblas_int       batch_count)
{
    return rocblas_gemv_strided_batched_impl(handle,
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
                                             rocblas_stride    strideA,
                                             const double*     x,
                                             rocblas_int       incx,
                                             rocblas_stride    stridex,
                                             const double*     beta,
                                             double*           y,
                                             rocblas_int       incy,
                                             rocblas_stride    stridey,
                                             rocblas_int       batch_count)
{
    return rocblas_gemv_strided_batched_impl(handle,
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

rocblas_status rocblas_cgemv_strided_batched(rocblas_handle               handle,
                                             rocblas_operation            transA,
                                             rocblas_int                  m,
                                             rocblas_int                  n,
                                             const rocblas_float_complex* alpha,
                                             const rocblas_float_complex* A,
                                             rocblas_int                  lda,
                                             rocblas_stride               strideA,
                                             const rocblas_float_complex* x,
                                             rocblas_int                  incx,
                                             rocblas_stride               stridex,
                                             const rocblas_float_complex* beta,
                                             rocblas_float_complex*       y,
                                             rocblas_int                  incy,
                                             rocblas_stride               stridey,
                                             rocblas_int                  batch_count)
{
    return rocblas_gemv_strided_batched_impl(handle,
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

rocblas_status rocblas_zgemv_strided_batched(rocblas_handle                handle,
                                             rocblas_operation             transA,
                                             rocblas_int                   m,
                                             rocblas_int                   n,
                                             const rocblas_double_complex* alpha,
                                             const rocblas_double_complex* A,
                                             rocblas_int                   lda,
                                             rocblas_stride                strideA,
                                             const rocblas_double_complex* x,
                                             rocblas_int                   incx,
                                             rocblas_stride                stridex,
                                             const rocblas_double_complex* beta,
                                             rocblas_double_complex*       y,
                                             rocblas_int                   incy,
                                             rocblas_stride                stridey,
                                             rocblas_int                   batch_count)
{
    return rocblas_gemv_strided_batched_impl(handle,
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
