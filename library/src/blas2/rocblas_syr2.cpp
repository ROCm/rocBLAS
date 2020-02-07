/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "rocblas_syr2.hpp"
#include "logging.h"
#include "utility.h"

namespace
{
    template <typename>
    constexpr char rocblas_syr2_name[] = "unknown";
    template <>
    constexpr char rocblas_syr2_name<float>[] = "rocblas_ssyr2";
    template <>
    constexpr char rocblas_syr2_name<double>[] = "rocblas_dsyr2";
    template <>
    constexpr char rocblas_syr2_name<rocblas_float_complex>[] = "rocblas_csyr2";
    template <>
    constexpr char rocblas_syr2_name<rocblas_double_complex>[] = "rocblas_zsyr2";

    template <typename T>
    rocblas_status rocblas_syr2_impl(rocblas_handle handle,
                                     rocblas_fill   uplo,
                                     rocblas_int    n,
                                     const T*       alpha,
                                     const T*       x,
                                     rocblas_int    incx,
                                     const T*       y,
                                     rocblas_int    incy,
                                     T*             A,
                                     rocblas_int    lda)
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
                              rocblas_syr2_name<T>,
                              uplo,
                              n,
                              log_trace_scalar_value(alpha),
                              x,
                              incx,
                              y,
                              incy,
                              A,
                              lda);

                if(layer_mode & rocblas_layer_mode_log_bench)
                    log_bench(handle,
                              "./rocblas-bench -f syr2 -r",
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
                              incy);
            }
            else
            {
                if(layer_mode & rocblas_layer_mode_log_trace)
                    log_trace(handle,
                              rocblas_syr2_name<T>,
                              uplo,
                              n,
                              log_trace_scalar_value(alpha),
                              x,
                              incx,
                              y,
                              incy,
                              A,
                              lda);
            }

            if(layer_mode & rocblas_layer_mode_log_profile)
                log_profile(handle,
                            rocblas_syr2_name<T>,
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

        if(uplo != rocblas_fill_lower && uplo != rocblas_fill_upper)
            return rocblas_status_invalid_value;
        if(n < 0 || !incx || !incy || lda < 1 || lda < n)
            return rocblas_status_invalid_size;
        if(!n)
            return rocblas_status_success;
        if(!x || !y || !A || !alpha)
            return rocblas_status_invalid_pointer;

        static constexpr rocblas_int    offset_x = 0, offset_y = 0, offset_A = 0, batch_count = 1;
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

rocblas_status rocblas_ssyr2(rocblas_handle handle,
                             rocblas_fill   uplo,
                             rocblas_int    n,
                             const float*   alpha,
                             const float*   x,
                             rocblas_int    incx,
                             const float*   y,
                             rocblas_int    incy,
                             float*         A,
                             rocblas_int    lda)
try
{
    return rocblas_syr2_impl(handle, uplo, n, alpha, x, incx, y, incy, A, lda);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_dsyr2(rocblas_handle handle,
                             rocblas_fill   uplo,
                             rocblas_int    n,
                             const double*  alpha,
                             const double*  x,
                             rocblas_int    incx,
                             const double*  y,
                             rocblas_int    incy,
                             double*        A,
                             rocblas_int    lda)
try
{
    return rocblas_syr2_impl(handle, uplo, n, alpha, x, incx, y, incy, A, lda);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_csyr2(rocblas_handle               handle,
                             rocblas_fill                 uplo,
                             rocblas_int                  n,
                             const rocblas_float_complex* alpha,
                             const rocblas_float_complex* x,
                             rocblas_int                  incx,
                             const rocblas_float_complex* y,
                             rocblas_int                  incy,
                             rocblas_float_complex*       A,
                             rocblas_int                  lda)
try
{
    return rocblas_syr2_impl(handle, uplo, n, alpha, x, incx, y, incy, A, lda);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_zsyr2(rocblas_handle                handle,
                             rocblas_fill                  uplo,
                             rocblas_int                   n,
                             const rocblas_double_complex* alpha,
                             const rocblas_double_complex* x,
                             rocblas_int                   incx,
                             const rocblas_double_complex* y,
                             rocblas_int                   incy,
                             rocblas_double_complex*       A,
                             rocblas_int                   lda)
try
{
    return rocblas_syr2_impl(handle, uplo, n, alpha, x, incx, y, incy, A, lda);
}
catch(...)
{
    return exception_to_rocblas_status();
}

} // extern "C"
