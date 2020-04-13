/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "rocblas_hemv.hpp"
#include "logging.h"

namespace
{
    template <typename>
    constexpr char rocblas_hemv_name[] = "unknown";
    template <>
    constexpr char rocblas_hemv_name<rocblas_float_complex>[] = "rocblas_chemv";
    template <>
    constexpr char rocblas_hemv_name<rocblas_double_complex>[] = "rocblas_zhemv";

    template <typename T>
    rocblas_status rocblas_hemv_impl(rocblas_handle handle,
                                     rocblas_fill   uplo,
                                     rocblas_int    n,
                                     const T*       alpha,
                                     const T*       A,
                                     rocblas_int    lda,
                                     const T*       x,
                                     rocblas_int    incx,
                                     const T*       beta,
                                     T*             y,
                                     rocblas_int    incy)
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
                              x,
                              incx,
                              log_trace_scalar_value(beta),
                              y,
                              incy);

                if(layer_mode & rocblas_layer_mode_log_bench)
                {
                    log_bench(handle,
                              "./rocblas-bench -f hemv -r",
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
                              LOG_BENCH_SCALAR_VALUE(beta),
                              "--incy",
                              incy);
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
                              x,
                              incx,
                              beta,
                              y,
                              incy);
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
                            "incx",
                            incx,
                            "incy",
                            incy);
        }

        if(n < 0 || lda < n || lda < 1 || !incx || !incy)
            return rocblas_status_invalid_size;

        if(!n)
            return rocblas_status_success;

        if(!A || !x || !y || !alpha || !beta)
            return rocblas_status_invalid_pointer;

        return rocblas_hemv_template(
            handle, uplo, n, alpha, A, 0, lda, 0, x, 0, incx, 0, beta, y, 0, incy, 0, 1);
    }
} // namespace

/*
* ===========================================================================
*    C wrapper
* ===========================================================================
*/

extern "C" {

rocblas_status rocblas_chemv(rocblas_handle               handle,
                             rocblas_fill                 uplo,
                             rocblas_int                  n,
                             const rocblas_float_complex* alpha,
                             const rocblas_float_complex* A,
                             rocblas_int                  lda,
                             const rocblas_float_complex* x,
                             rocblas_int                  incx,
                             const rocblas_float_complex* beta,
                             rocblas_float_complex*       y,
                             rocblas_int                  incy)
try
{
    return rocblas_hemv_impl(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_zhemv(rocblas_handle                handle,
                             rocblas_fill                  uplo,
                             rocblas_int                   n,
                             const rocblas_double_complex* alpha,
                             const rocblas_double_complex* A,
                             rocblas_int                   lda,
                             const rocblas_double_complex* x,
                             rocblas_int                   incx,
                             const rocblas_double_complex* beta,
                             rocblas_double_complex*       y,
                             rocblas_int                   incy)
try
{
    return rocblas_hemv_impl(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy);
}
catch(...)
{
    return exception_to_rocblas_status();
}

} // extern "C"
