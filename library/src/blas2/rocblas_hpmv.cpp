/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "rocblas_hpmv.hpp"
#include "logging.h"

namespace
{
    template <typename>
    constexpr char rocblas_hpmv_name[] = "unknown";
    template <>
    constexpr char rocblas_hpmv_name<rocblas_float_complex>[] = "rocblas_chpmv";
    template <>
    constexpr char rocblas_hpmv_name<rocblas_double_complex>[] = "rocblas_zhpmv";

    template <typename T>
    rocblas_status rocblas_hpmv_impl(rocblas_handle handle,
                                     rocblas_fill   uplo,
                                     rocblas_int    n,
                                     const T*       alpha,
                                     const T*       AP,
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
                              rocblas_hpmv_name<T>,
                              uplo,
                              n,
                              log_trace_scalar_value(alpha),
                              AP,
                              x,
                              incx,
                              log_trace_scalar_value(beta),
                              y,
                              incy);

                if(layer_mode & rocblas_layer_mode_log_bench)
                {
                    log_bench(handle,
                              "./rocblas-bench -f hpmv -r",
                              rocblas_precision_string<T>,
                              "--uplo",
                              uplo_letter,
                              "-n",
                              n,
                              LOG_BENCH_SCALAR_VALUE(alpha),
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
                    log_trace(
                        handle, rocblas_hpmv_name<T>, uplo, n, alpha, AP, x, incx, beta, y, incy);
            }

            if(layer_mode & rocblas_layer_mode_log_profile)
                log_profile(handle,
                            rocblas_hpmv_name<T>,
                            "uplo",
                            uplo_letter,
                            "N",
                            n,
                            "incx",
                            incx,
                            "incy",
                            incy);
        }

        if(uplo != rocblas_fill_lower && uplo != rocblas_fill_upper)
            return rocblas_status_invalid_value;

        if(n < 0 || !incx || !incy)
            return rocblas_status_invalid_size;

        if(!n)
            return rocblas_status_success;

        if(!AP || !x || !y || !alpha || !beta)
            return rocblas_status_invalid_pointer;

        constexpr rocblas_int    offset_A = 0, offset_x = 0, offset_y = 0, batch_count = 1;
        constexpr rocblas_stride stride_A = 0, stride_x = 0, stride_y = 0;
        return rocblas_hpmv_template(handle,
                                     uplo,
                                     n,
                                     alpha,
                                     AP,
                                     offset_A,
                                     stride_A,
                                     x,
                                     offset_x,
                                     incx,
                                     stride_x,
                                     beta,
                                     y,
                                     offset_y,
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

rocblas_status rocblas_chpmv(rocblas_handle               handle,
                             rocblas_fill                 uplo,
                             rocblas_int                  n,
                             const rocblas_float_complex* alpha,
                             const rocblas_float_complex* AP,
                             const rocblas_float_complex* x,
                             rocblas_int                  incx,
                             const rocblas_float_complex* beta,
                             rocblas_float_complex*       y,
                             rocblas_int                  incy)
try
{
    return rocblas_hpmv_impl(handle, uplo, n, alpha, AP, x, incx, beta, y, incy);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_zhpmv(rocblas_handle                handle,
                             rocblas_fill                  uplo,
                             rocblas_int                   n,
                             const rocblas_double_complex* alpha,
                             const rocblas_double_complex* AP,
                             const rocblas_double_complex* x,
                             rocblas_int                   incx,
                             const rocblas_double_complex* beta,
                             rocblas_double_complex*       y,
                             rocblas_int                   incy)
try
{
    return rocblas_hpmv_impl(handle, uplo, n, alpha, AP, x, incx, beta, y, incy);
}
catch(...)
{
    return exception_to_rocblas_status();
}

} // extern "C"
