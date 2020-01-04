/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "rocblas_swap.hpp"
#include "logging.h"
#include "utility.h"

namespace
{
    template <typename>
    constexpr char rocblas_swap_name[] = "unknown";
    template <>
    constexpr char rocblas_swap_name<float>[] = "rocblas_sswap";
    template <>
    constexpr char rocblas_swap_name<double>[] = "rocblas_dswap";
    template <>
    constexpr char rocblas_swap_name<rocblas_half>[] = "rocblas_hswap";
    template <>
    constexpr char rocblas_swap_name<rocblas_float_complex>[] = "rocblas_cswap";
    template <>
    constexpr char rocblas_swap_name<rocblas_double_complex>[] = "rocblas_zswap";

    template <rocblas_int NB, class T>
    rocblas_status rocblas_swap_impl(
        rocblas_handle handle, rocblas_int n, T* x, rocblas_int incx, T* y, rocblas_int incy)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);

        auto layer_mode = handle->layer_mode;
        if(layer_mode & rocblas_layer_mode_log_trace)
            log_trace(handle, rocblas_swap_name<T>, n, x, incx, y, incy);
        if(layer_mode & rocblas_layer_mode_log_bench)
            log_bench(handle,
                      "./rocblas-bench -f swap -r",
                      rocblas_precision_string<T>,
                      "-n",
                      n,
                      "--incx",
                      incx,
                      "--incy",
                      incy);
        if(layer_mode & rocblas_layer_mode_log_profile)
            log_profile(handle, rocblas_swap_name<T>, "N", n, "incx", incx, "incy", incy);

        if(n <= 0)
            return rocblas_status_success;

        if(!x || !y)
            return rocblas_status_invalid_pointer;

        return rocblas_swap_template<NB>(handle, n, x, 0, incx, 0, y, 0, incy, 0);
    }

} // namespace

/* ============================================================================================ */

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocblas_sswap(
    rocblas_handle handle, rocblas_int n, float* x, rocblas_int incx, float* y, rocblas_int incy)
try
{
    constexpr rocblas_int NB = 256;
    return rocblas_swap_impl<NB>(handle, n, x, incx, y, incy);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_dswap(
    rocblas_handle handle, rocblas_int n, double* x, rocblas_int incx, double* y, rocblas_int incy)
try
{
    constexpr rocblas_int NB = 256;
    return rocblas_swap_impl<NB>(handle, n, x, incx, y, incy);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_cswap(rocblas_handle         handle,
                             rocblas_int            n,
                             rocblas_float_complex* x,
                             rocblas_int            incx,
                             rocblas_float_complex* y,
                             rocblas_int            incy)
try
{
    constexpr rocblas_int NB = 256;
    return rocblas_swap_impl<NB>(handle, n, x, incx, y, incy);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_zswap(rocblas_handle          handle,
                             rocblas_int             n,
                             rocblas_double_complex* x,
                             rocblas_int             incx,
                             rocblas_double_complex* y,
                             rocblas_int             incy)
try
{
    constexpr rocblas_int NB = 256;
    return rocblas_swap_impl<NB>(handle, n, x, incx, y, incy);
}
catch(...)
{
    return exception_to_rocblas_status();
}

} // extern "C"
