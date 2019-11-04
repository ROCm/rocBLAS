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
    constexpr char rocblas_gemv_name<float>[] = "rocblas_sgemv_batched";
    template <>
    constexpr char rocblas_gemv_name<double>[] = "rocblas_dgemv_batched";
    template <>
    constexpr char rocblas_gemv_name<rocblas_float_complex>[] = "rocblas_cgemv_batched";
    template <>
    constexpr char rocblas_gemv_name<rocblas_double_complex>[] = "rocblas_zgemv_batched";

    template <typename T>
    rocblas_status rocblas_gemv_batched_impl(rocblas_handle    handle,
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
                              x,
                              incx,
                              log_trace_scalar_value(beta),
                              y,
                              incy,
                              batch_count);

                if(layer_mode & rocblas_layer_mode_log_bench)
                {
                    log_bench(handle,
                              "./rocblas-bench -f gemv_batched -r",
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
                              "--incx",
                              incx,
                              LOG_BENCH_SCALAR_VALUE(beta),
                              "--incy",
                              incy,
                              "--batch",
                              batch_count);
                }
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
                              x,
                              incx,
                              beta,
                              y,
                              incy,
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
                            "incx",
                            incx,
                            "incy",
                            incy,
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
                                        0,
                                        x,
                                        0,
                                        incx,
                                        0,
                                        beta,
                                        y,
                                        0,
                                        incy,
                                        0,
                                        batch_count);
    }
} // namespace

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocblas_sgemv_batched(rocblas_handle     handle,
                                     rocblas_operation  transA,
                                     rocblas_int        m,
                                     rocblas_int        n,
                                     const float*       alpha,
                                     const float* const A[],
                                     rocblas_int        lda,
                                     const float* const x[],
                                     rocblas_int        incx,
                                     const float*       beta,
                                     float* const       y[],
                                     rocblas_int        incy,
                                     rocblas_int        batch_count)
{
    return rocblas_gemv_batched_impl(
        handle, transA, m, n, alpha, A, lda, x, incx, beta, y, incy, batch_count);
}

rocblas_status rocblas_dgemv_batched(rocblas_handle      handle,
                                     rocblas_operation   transA,
                                     rocblas_int         m,
                                     rocblas_int         n,
                                     const double*       alpha,
                                     const double* const A[],
                                     rocblas_int         lda,
                                     const double* const x[],
                                     rocblas_int         incx,
                                     const double*       beta,
                                     double* const       y[],
                                     rocblas_int         incy,
                                     rocblas_int         batch_count)
{
    return rocblas_gemv_batched_impl(
        handle, transA, m, n, alpha, A, lda, x, incx, beta, y, incy, batch_count);
}

rocblas_status rocblas_cgemv_batched(rocblas_handle                     handle,
                                     rocblas_operation                  transA,
                                     rocblas_int                        m,
                                     rocblas_int                        n,
                                     const rocblas_float_complex*       alpha,
                                     const rocblas_float_complex* const A[],
                                     rocblas_int                        lda,
                                     const rocblas_float_complex* const x[],
                                     rocblas_int                        incx,
                                     const rocblas_float_complex*       beta,
                                     rocblas_float_complex* const       y[],
                                     rocblas_int                        incy,
                                     rocblas_int                        batch_count)
{
    return rocblas_gemv_batched_impl(
        handle, transA, m, n, alpha, A, lda, x, incx, beta, y, incy, batch_count);
}

rocblas_status rocblas_zgemv_batched(rocblas_handle                      handle,
                                     rocblas_operation                   transA,
                                     rocblas_int                         m,
                                     rocblas_int                         n,
                                     const rocblas_double_complex*       alpha,
                                     const rocblas_double_complex* const A[],
                                     rocblas_int                         lda,
                                     const rocblas_double_complex* const x[],
                                     rocblas_int                         incx,
                                     const rocblas_double_complex*       beta,
                                     rocblas_double_complex* const       y[],
                                     rocblas_int                         incy,
                                     rocblas_int                         batch_count)
{
    return rocblas_gemv_batched_impl(
        handle, transA, m, n, alpha, A, lda, x, incx, beta, y, incy, batch_count);
}

} // extern "C"
