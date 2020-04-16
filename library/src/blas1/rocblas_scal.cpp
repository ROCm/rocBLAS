/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "rocblas_scal.hpp"
#include "handle.h"
#include "logging.h"
#include "rocblas.h"
#include "utility.h"

namespace
{
    template <typename T, typename = T>
    constexpr char rocblas_scal_name[] = "unknown";
    template <>
    constexpr char rocblas_scal_name<float>[] = "rocblas_sscal";
    template <>
    constexpr char rocblas_scal_name<double>[] = "rocblas_dscal";
    template <>
    constexpr char rocblas_scal_name<rocblas_float_complex>[] = "rocblas_cscal";
    template <>
    constexpr char rocblas_scal_name<rocblas_double_complex>[] = "rocblas_zscal";
    template <>
    constexpr char rocblas_scal_name<rocblas_float_complex, float>[] = "rocblas_csscal";
    template <>
    constexpr char rocblas_scal_name<rocblas_double_complex, double>[] = "rocblas_zdscal";

    template <rocblas_int NB, typename T, typename U>
    rocblas_status rocblas_scal_impl(
        rocblas_handle handle, rocblas_int n, const U* alpha, T* x, rocblas_int incx)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        auto layer_mode = handle->layer_mode;
        if(handle->pointer_mode == rocblas_pointer_mode_host)
        {
            if(layer_mode & rocblas_layer_mode_log_trace)
                log_trace(handle, rocblas_scal_name<T, U>, n, *alpha, x, incx);

            // there are an extra 2 scal functions, thus
            // the -r mode will not work correctly. Substitute
            // with --a_type and --b_type (?)
            // ANSWER: -r is syntatic sugar; the types can be specified separately
            if(layer_mode & rocblas_layer_mode_log_bench)
            {
                rocblas_ostream alphass;
                alphass << "--alpha " << std::real(*alpha)
                        << (std::imag(*alpha) != 0
                                ? (" --alphai " + std::to_string(std::imag(*alpha)))
                                : "");
                log_bench(handle,
                          "./rocblas-bench -f scal --a_type",
                          rocblas_precision_string<T>,
                          "--b_type",
                          rocblas_precision_string<U>,
                          "-n",
                          n,
                          "--incx",
                          incx,
                          alphass.str());
            }
        }
        else
        {
            if(layer_mode & rocblas_layer_mode_log_trace)
                log_trace(handle, rocblas_scal_name<T, U>, n, alpha, x, incx);
        }
        if(layer_mode & rocblas_layer_mode_log_profile)
            log_profile(handle, rocblas_scal_name<T, U>, "N", n, "incx", incx);

        if(n <= 0 || incx <= 0)
            return rocblas_status_success;
        if(!x || !alpha)
            return rocblas_status_invalid_pointer;

        RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);

        return rocblas_scal_template<NB, T>(handle, n, alpha, 0, x, 0, incx, 0, 1);
    }
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocblas_sscal(
    rocblas_handle handle, rocblas_int n, const float* alpha, float* x, rocblas_int incx)
try
{
    constexpr rocblas_int NB = 256;
    return rocblas_scal_impl<NB>(handle, n, alpha, x, incx);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_dscal(
    rocblas_handle handle, rocblas_int n, const double* alpha, double* x, rocblas_int incx)
try
{
    constexpr rocblas_int NB = 256;
    return rocblas_scal_impl<NB>(handle, n, alpha, x, incx);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_cscal(rocblas_handle               handle,
                             rocblas_int                  n,
                             const rocblas_float_complex* alpha,
                             rocblas_float_complex*       x,
                             rocblas_int                  incx)
try
{
    constexpr rocblas_int NB = 256;
    return rocblas_scal_impl<NB>(handle, n, alpha, x, incx);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_zscal(rocblas_handle                handle,
                             rocblas_int                   n,
                             const rocblas_double_complex* alpha,
                             rocblas_double_complex*       x,
                             rocblas_int                   incx)
try
{
    constexpr rocblas_int NB = 256;
    return rocblas_scal_impl<NB>(handle, n, alpha, x, incx);
}
catch(...)
{
    return exception_to_rocblas_status();
}

// Scal with a real alpha & complex vector
rocblas_status rocblas_csscal(rocblas_handle         handle,
                              rocblas_int            n,
                              const float*           alpha,
                              rocblas_float_complex* x,
                              rocblas_int            incx)
try
{
    constexpr rocblas_int NB = 256;
    return rocblas_scal_impl<NB>(handle, n, alpha, x, incx);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_zdscal(rocblas_handle          handle,
                              rocblas_int             n,
                              const double*           alpha,
                              rocblas_double_complex* x,
                              rocblas_int             incx)
try
{
    constexpr rocblas_int NB = 256;
    return rocblas_scal_impl<NB>(handle, n, alpha, x, incx);
}
catch(...)
{
    return exception_to_rocblas_status();
}

} // extern "C"
