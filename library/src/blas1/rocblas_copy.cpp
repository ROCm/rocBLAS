/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "rocblas_copy.hpp"
#include "handle.h"
#include "logging.h"
#include "rocblas.h"
#include "utility.h"

namespace
{
    template <typename>
    constexpr char rocblas_copy_name[] = "unknown";
    template <>
    constexpr char rocblas_copy_name<float>[] = "rocblas_scopy";
    template <>
    constexpr char rocblas_copy_name<double>[] = "rocblas_dcopy";
    template <>
    constexpr char rocblas_copy_name<rocblas_half>[] = "rocblas_hcopy";
    template <>
    constexpr char rocblas_copy_name<rocblas_float_complex>[] = "rocblas_ccopy";
    template <>
    constexpr char rocblas_copy_name<rocblas_double_complex>[] = "rocblas_zcopy";

    template <rocblas_int NB, typename T>
    rocblas_status rocblas_copy_impl(
        rocblas_handle handle, rocblas_int n, const T* x, rocblas_int incx, T* y, rocblas_int incy)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);

        auto layer_mode = handle->layer_mode;
        if(layer_mode & rocblas_layer_mode_log_trace)
            log_trace(handle, rocblas_copy_name<T>, n, x, incx, y, incy);

        if(layer_mode & rocblas_layer_mode_log_bench)
            log_bench(handle,
                      "./rocblas-bench -f copy -r",
                      rocblas_precision_string<T>,
                      "-n",
                      n,
                      "--incx",
                      incx,
                      "--incy",
                      incy);

        if(layer_mode & rocblas_layer_mode_log_profile)
            log_profile(handle, rocblas_copy_name<T>, "N", n, "incx", incx, "incy", incy);

        return rocblas_copy_template<NB, T>(handle, n, x, 0, incx, 0, y, 0, incy, 0, 1);
    }

} // namespace

/* ============================================================================================ */

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocblas_scopy(rocblas_handle handle,
                             rocblas_int    n,
                             const float*   x,
                             rocblas_int    incx,
                             float*         y,
                             rocblas_int    incy)
{
    constexpr int NB = 256;
    return rocblas_copy_impl<NB>(handle, n, x, incx, y, incy);
}

rocblas_status rocblas_dcopy(rocblas_handle handle,
                             rocblas_int    n,
                             const double*  x,
                             rocblas_int    incx,
                             double*        y,
                             rocblas_int    incy)
{
    constexpr int NB = 256;
    return rocblas_copy_impl<NB>(handle, n, x, incx, y, incy);
}

rocblas_status rocblas_hcopy(rocblas_handle      handle,
                             rocblas_int         n,
                             const rocblas_half* x,
                             rocblas_int         incx,
                             rocblas_half*       y,
                             rocblas_int         incy)
{
    constexpr int NB = 256;
    return rocblas_copy_impl<NB>(handle, n, x, incx, y, incy);
}

rocblas_status rocblas_ccopy(rocblas_handle               handle,
                             rocblas_int                  n,
                             const rocblas_float_complex* x,
                             rocblas_int                  incx,
                             rocblas_float_complex*       y,
                             rocblas_int                  incy)
{
    constexpr int NB = 256;
    return rocblas_copy_impl<NB>(handle, n, x, incx, y, incy);
}

rocblas_status rocblas_zcopy(rocblas_handle                handle,
                             rocblas_int                   n,
                             const rocblas_double_complex* x,
                             rocblas_int                   incx,
                             rocblas_double_complex*       y,
                             rocblas_int                   incy)
{
    constexpr int NB = 256;
    return rocblas_copy_impl<NB>(handle, n, x, incx, y, incy);
}

} // extern "C"
