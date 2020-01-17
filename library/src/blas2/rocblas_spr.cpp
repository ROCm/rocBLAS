/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "rocblas_spr.hpp"
#include "logging.h"
#include "utility.h"

namespace
{
    template <typename>
    constexpr char rocblas_spr_name[] = "unknown";
    template <>
    constexpr char rocblas_spr_name<float>[] = "rocblas_sspr";
    template <>
    constexpr char rocblas_spr_name<double>[] = "rocblas_dspr";

    template <typename T>
    rocblas_status rocblas_spr_impl(rocblas_handle handle,
                                    rocblas_fill   uplo,
                                    rocblas_int    n,
                                    const T*       alpha,
                                    const T*       x,
                                    rocblas_int    incx,
                                    T*             AP)
    {
        if(!handle)
            return rocblas_status_invalid_handle;
        RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);
        if(!alpha)
            return rocblas_status_invalid_pointer;
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
                              rocblas_spr_name<T>,
                              uplo,
                              n,
                              log_trace_scalar_value(alpha),
                              x,
                              incx,
                              AP);

                if(layer_mode & rocblas_layer_mode_log_bench)
                    log_bench(handle,
                              "./rocblas-bench -f spr -r",
                              rocblas_precision_string<T>,
                              "--uplo",
                              uplo_letter,
                              "-n",
                              n,
                              LOG_BENCH_SCALAR_VALUE(alpha),
                              "--incx",
                              incx);
            }
            else
            {
                if(layer_mode & rocblas_layer_mode_log_trace)
                    log_trace(handle,
                              rocblas_spr_name<T>,
                              uplo,
                              n,
                              log_trace_scalar_value(alpha),
                              x,
                              incx,
                              AP);
            }

            if(layer_mode & rocblas_layer_mode_log_profile)
                log_profile(handle, rocblas_spr_name<T>, "uplo", uplo_letter, "N", n, "incx", incx);
        }

        if(uplo != rocblas_fill_lower && uplo != rocblas_fill_upper)
            return rocblas_status_not_implemented;
        if(n < 0 || !incx)
            return rocblas_status_invalid_size;
        if(!n)
            return rocblas_status_success;
        if(!x || !AP)
            return rocblas_status_invalid_pointer;

        return rocblas_spr_template(handle, uplo, n, alpha, x, 0, incx, 0, AP, 0, 0, 1);
    }

}
/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocblas_sspr(rocblas_handle handle,
                            rocblas_fill   uplo,
                            rocblas_int    n,
                            const float*   alpha,
                            const float*   x,
                            rocblas_int    incx,
                            float*         AP)
try
{
    return rocblas_spr_impl(handle, uplo, n, alpha, x, incx, AP);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_dspr(rocblas_handle handle,
                            rocblas_fill   uplo,
                            rocblas_int    n,
                            const double*  alpha,
                            const double*  x,
                            rocblas_int    incx,
                            double*        AP)
try
{
    return rocblas_spr_impl(handle, uplo, n, alpha, x, incx, AP);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_cspr(rocblas_handle               handle,
                            rocblas_fill                 uplo,
                            rocblas_int                  n,
                            const rocblas_float_complex* alpha,
                            const rocblas_float_complex* x,
                            rocblas_int                  incx,
                            rocblas_float_complex*       AP)
try
{
    return rocblas_spr_impl(handle, uplo, n, alpha, x, incx, AP);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_zspr(rocblas_handle                handle,
                            rocblas_fill                  uplo,
                            rocblas_int                   n,
                            const rocblas_double_complex* alpha,
                            const rocblas_double_complex* x,
                            rocblas_int                   incx,
                            rocblas_double_complex*       AP)
try
{
    return rocblas_spr_impl(handle, uplo, n, alpha, x, incx, AP);
}
catch(...)
{
    return exception_to_rocblas_status();
}

} // extern "C"
