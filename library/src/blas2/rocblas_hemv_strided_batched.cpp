/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "handle.h"
#include "logging.h"
#include "rocblas.h"
#include "rocblas_hemv.hpp"
#include "utility.h"
#include <limits>

namespace
{
    template <typename>
    constexpr char rocblas_hemv_name[] = "unknown";
    template <>
    constexpr char rocblas_hemv_name<rocblas_float_complex>[] = "rocblas_chemv_strided_batched";
    template <>
    constexpr char rocblas_hemv_name<rocblas_double_complex>[] = "rocblas_zhemv_strided_batched";

    template <typename T>
    rocblas_status rocblas_hemv_strided_batched_impl(rocblas_handle handle,
                                                     rocblas_fill   uplo,
                                                     rocblas_int    n,
                                                     const T*       alpha,
                                                     const T*       A,
                                                     rocblas_int    lda,
                                                     rocblas_stride stride_A,
                                                     const T*       x,
                                                     rocblas_int    incx,
                                                     rocblas_stride stride_x,
                                                     const T*       beta,
                                                     T*             y,
                                                     rocblas_int    incy,
                                                     rocblas_stride stride_y,
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
                              rocblas_hemv_name<T>,
                              uplo,
                              n,
                              log_trace_scalar_value(alpha),
                              A,
                              lda,
                              stride_A,
                              x,
                              incx,
                              stride_x,
                              log_trace_scalar_value(beta),
                              y,
                              incy,
                              stride_y,
                              batch_count);

                if(layer_mode & rocblas_layer_mode_log_bench)
                {
                    log_bench(handle,
                              "./rocblas-bench -f hemv_strided_batched -r",
                              rocblas_precision_string<T>,
                              "--uplo",
                              uplo_letter,
                              "-n",
                              n,
                              LOG_BENCH_SCALAR_VALUE(alpha),
                              "--lda",
                              lda,
                              "--stride_a",
                              stride_A,
                              "--incx",
                              incx,
                              "--stride_x",
                              stride_x,
                              LOG_BENCH_SCALAR_VALUE(beta),
                              "--incy",
                              incy,
                              "--stride_y",
                              stride_y,
                              "--batch_count",
                              batch_count);
                }
            }
            else
            {
                if(layer_mode & rocblas_layer_mode_log_trace)
                    log_trace(handle,
                              rocblas_hemv_name<T>,
                              uplo,
                              n,
                              alpha,
                              A,
                              lda,
                              stride_A,
                              x,
                              incx,
                              stride_x,
                              beta,
                              y,
                              incy,
                              stride_y,
                              batch_count);
            }

            if(layer_mode & rocblas_layer_mode_log_profile)
                log_profile(handle,
                            rocblas_hemv_name<T>,
                            "uplo",
                            uplo_letter,
                            "N",
                            n,
                            "lda",
                            lda,
                            "stride_a",
                            stride_A,
                            "incx",
                            incx,
                            "stride_x",
                            stride_x,
                            "incy",
                            incy,
                            "stride_y",
                            stride_y,
                            "batch_count",
                            batch_count);
        }

        if(n < 0 || lda < n || lda < 1 || !incx || !incy || batch_count < 0)
            return rocblas_status_invalid_size;

        if(!n || !batch_count)
            return rocblas_status_success;

        if(!A || !x || !y || !alpha || !beta)
            return rocblas_status_invalid_pointer;

        return rocblas_hemv_template(handle,
                                     uplo,
                                     n,
                                     alpha,
                                     A,
                                     0,
                                     lda,
                                     stride_A,
                                     x,
                                     0,
                                     incx,
                                     stride_x,
                                     beta,
                                     y,
                                     0,
                                     incy,
                                     stride_y,
                                     batch_count);
    }

} // namespace

/*
* ===========================================================================
*    C wrapper
* ===========================================================================
*/

extern "C" {

rocblas_status rocblas_chemv_strided_batched(rocblas_handle               handle,
                                             rocblas_fill                 uplo,
                                             rocblas_int                  n,
                                             const rocblas_float_complex* alpha,
                                             const rocblas_float_complex* A,
                                             rocblas_int                  lda,
                                             rocblas_stride               stride_A,
                                             const rocblas_float_complex* x,
                                             rocblas_int                  incx,
                                             rocblas_stride               stride_x,
                                             const rocblas_float_complex* beta,
                                             rocblas_float_complex*       y,
                                             rocblas_int                  incy,
                                             rocblas_stride               stride_y,
                                             rocblas_int                  batch_count)
try
{
    return rocblas_hemv_strided_batched_impl(handle,
                                             uplo,
                                             n,
                                             alpha,
                                             A,
                                             lda,
                                             stride_A,
                                             x,
                                             incx,
                                             stride_x,
                                             beta,
                                             y,
                                             incy,
                                             stride_y,
                                             batch_count);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_zhemv_strided_batched(rocblas_handle                handle,
                                             rocblas_fill                  uplo,
                                             rocblas_int                   n,
                                             const rocblas_double_complex* alpha,
                                             const rocblas_double_complex* A,
                                             rocblas_int                   lda,
                                             rocblas_stride                stride_A,
                                             const rocblas_double_complex* x,
                                             rocblas_int                   incx,
                                             rocblas_stride                stride_x,
                                             const rocblas_double_complex* beta,
                                             rocblas_double_complex*       y,
                                             rocblas_int                   incy,
                                             rocblas_stride                stride_y,
                                             rocblas_int                   batch_count)
try
{
    return rocblas_hemv_strided_batched_impl(handle,
                                             uplo,
                                             n,
                                             alpha,
                                             A,
                                             lda,
                                             stride_A,
                                             x,
                                             incx,
                                             stride_x,
                                             beta,
                                             y,
                                             incy,
                                             stride_y,
                                             batch_count);
}
catch(...)
{
    return exception_to_rocblas_status();
}

} // extern "C"
