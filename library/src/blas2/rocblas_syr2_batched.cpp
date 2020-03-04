/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "logging.h"
#include "rocblas_syr2.hpp"
#include "utility.h"

namespace
{
    template <typename>
    constexpr char rocblas_syr2_batched_name[] = "unknown";
    template <>
    constexpr char rocblas_syr2_batched_name<float>[] = "rocblas_ssyr2_batched";
    template <>
    constexpr char rocblas_syr2_batched_name<double>[] = "rocblas_dsyr2_batched";
    template <>
    constexpr char rocblas_syr2_batched_name<rocblas_float_complex>[] = "rocblas_csyr2_batched";
    template <>
    constexpr char rocblas_syr2_batched_name<rocblas_double_complex>[] = "rocblas_zsyr2_batched";

    template <typename T>
    rocblas_status rocblas_syr2_batched_impl(rocblas_handle handle,
                                             rocblas_fill   uplo,
                                             rocblas_int    n,
                                             const T*       alpha,
                                             const T* const x[],
                                             rocblas_int    incx,
                                             const T* const y[],
                                             rocblas_int    incy,
                                             T* const       A[],
                                             rocblas_int    lda,
                                             rocblas_int    batch_count)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);

        auto layer_mode = handle->layer_mode;
        if(layer_mode
           & (rocblas_layer_mode_log_trace | rocblas_layer_mode_log_bench
              | rocblas_layer_mode_log_profile))
        {
            auto uplo_letter = rocblas_fill_letter(uplo);

            if(handle->pointer_mode == rocblas_pointer_mode_host)
            {
                if(layer_mode & rocblas_layer_mode_log_trace)
                    log_trace(handle,
                              rocblas_syr2_batched_name<T>,
                              uplo,
                              n,
                              log_trace_scalar_value(alpha),
                              0,
                              x,
                              incx,
                              y,
                              incy,
                              A,
                              lda);

                if(layer_mode & rocblas_layer_mode_log_bench)
                    log_bench(handle,
                              "./rocblas-bench -f syr2_batched -r",
                              rocblas_precision_string<T>,
                              "--uplo",
                              uplo_letter,
                              "-n",
                              n,
                              LOG_BENCH_SCALAR_VALUE(alpha),
                              "--lda",
                              lda,
                              "--incx",
                              incx,
                              "--incy",
                              incy,
                              "--batch_count",
                              batch_count);
            }
            else
            {
                if(layer_mode & rocblas_layer_mode_log_trace)
                    log_trace(handle,
                              rocblas_syr2_batched_name<T>,
                              uplo,
                              n,
                              log_trace_scalar_value(alpha),
                              x,
                              incx,
                              y,
                              incy,
                              A,
                              lda,
                              batch_count);
            }

            if(layer_mode & rocblas_layer_mode_log_profile)
                log_profile(handle,
                            rocblas_syr2_batched_name<T>,
                            "uplo",
                            uplo_letter,
                            "N",
                            n,
                            "incx",
                            incx,
                            "incy",
                            incy,
                            "lda",
                            lda,
                            "batch_count",
                            batch_count);
        }

        if(uplo != rocblas_fill_lower && uplo != rocblas_fill_upper)
            return rocblas_status_invalid_value;
        if(n < 0 || !incx || !incy || batch_count < 0 || lda < 1 || lda < n)
            return rocblas_status_invalid_size;
        if(!n || !batch_count)
            return rocblas_status_success;
        if(!x || !y || !A || !alpha)
            return rocblas_status_invalid_pointer;

        static constexpr rocblas_int    offset_x = 0, offset_y = 0, offset_A = 0;
        static constexpr rocblas_stride stride_x = 0, stride_y = 0, stride_A = 0;
        return rocblas_syr2_template(handle,
                                     uplo,
                                     n,
                                     alpha,
                                     x,
                                     offset_x,
                                     incx,
                                     stride_x,
                                     y,
                                     offset_y,
                                     incy,
                                     stride_y,
                                     A,
                                     lda,
                                     offset_A,
                                     stride_A,
                                     batch_count);
    }

}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocblas_ssyr2_batched(rocblas_handle     handle,
                                     rocblas_fill       uplo,
                                     rocblas_int        n,
                                     const float*       alpha,
                                     const float* const x[],
                                     rocblas_int        incx,
                                     const float* const y[],
                                     rocblas_int        incy,
                                     float* const       A[],
                                     rocblas_int        lda,
                                     rocblas_int        batch_count)
try
{
    return rocblas_syr2_batched_impl(handle, uplo, n, alpha, x, incx, y, incy, A, lda, batch_count);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_dsyr2_batched(rocblas_handle      handle,
                                     rocblas_fill        uplo,
                                     rocblas_int         n,
                                     const double*       alpha,
                                     const double* const x[],
                                     rocblas_int         incx,
                                     const double* const y[],
                                     rocblas_int         incy,
                                     double* const       A[],
                                     rocblas_int         lda,
                                     rocblas_int         batch_count)
try
{
    return rocblas_syr2_batched_impl(handle, uplo, n, alpha, x, incx, y, incy, A, lda, batch_count);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_csyr2_batched(rocblas_handle                     handle,
                                     rocblas_fill                       uplo,
                                     rocblas_int                        n,
                                     const rocblas_float_complex*       alpha,
                                     const rocblas_float_complex* const x[],
                                     rocblas_int                        incx,
                                     const rocblas_float_complex* const y[],
                                     rocblas_int                        incy,
                                     rocblas_float_complex* const       A[],
                                     rocblas_int                        lda,
                                     rocblas_int                        batch_count)
try
{
    return rocblas_syr2_batched_impl(handle, uplo, n, alpha, x, incx, y, incy, A, lda, batch_count);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_zsyr2_batched(rocblas_handle                      handle,
                                     rocblas_fill                        uplo,
                                     rocblas_int                         n,
                                     const rocblas_double_complex*       alpha,
                                     const rocblas_double_complex* const x[],
                                     rocblas_int                         incx,
                                     const rocblas_double_complex* const y[],
                                     rocblas_int                         incy,
                                     rocblas_double_complex* const       A[],
                                     rocblas_int                         lda,
                                     rocblas_int                         batch_count)
try
{
    return rocblas_syr2_batched_impl(handle, uplo, n, alpha, x, incx, y, incy, A, lda, batch_count);
}
catch(...)
{
    return exception_to_rocblas_status();
}

} // extern "C"
