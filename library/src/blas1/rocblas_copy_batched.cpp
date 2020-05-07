/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "handle.h"
#include "logging.h"
#include "rocblas.h"
#include "rocblas_copy.hpp"
#include "utility.h"

namespace
{
    template <typename>
    constexpr char rocblas_copy_batched_name[] = "unknown";
    template <>
    constexpr char rocblas_copy_batched_name<float>[] = "rocblas_scopy_batched";
    template <>
    constexpr char rocblas_copy_batched_name<double>[] = "rocblas_dcopy_batched";
    template <>
    constexpr char rocblas_copy_batched_name<rocblas_half>[] = "rocblas_hcopy_batched";
    template <>
    constexpr char rocblas_copy_batched_name<rocblas_float_complex>[] = "rocblas_ccopy_batched";
    template <>
    constexpr char rocblas_copy_batched_name<rocblas_double_complex>[] = "rocblas_zcopy_batched";

    template <rocblas_int NB, typename T>
    rocblas_status rocblas_copy_batched_impl(rocblas_handle handle,
                                             rocblas_int    n,
                                             const T* const x[],
                                             rocblas_int    incx,
                                             T* const       y[],
                                             rocblas_int    incy,
                                             rocblas_int    batch_count)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);

        auto layer_mode = handle->layer_mode;
        if(layer_mode & rocblas_layer_mode_log_trace)
            log_trace(handle, rocblas_copy_batched_name<T>, n, x, incx, y, incy, batch_count);

        if(layer_mode & rocblas_layer_mode_log_bench)
            log_bench(handle,
                      "./rocblas-bench -f copy_batched -r",
                      rocblas_precision_string<T>,
                      "-n",
                      n,
                      "--incx",
                      incx,
                      "--incy",
                      incy,
                      "batch_count",
                      batch_count);

        if(layer_mode & rocblas_layer_mode_log_profile)
            log_profile(handle,
                        rocblas_copy_batched_name<T>,
                        "N",
                        n,
                        "incx",
                        incx,
                        "incy",
                        incy,
                        "batch_count",
                        batch_count);

        if(n <= 0 || batch_count <= 0)
            return rocblas_status_success;
        if(!x || !y)
            return rocblas_status_invalid_pointer;

        return rocblas_copy_template<false, NB>(
            handle, n, x, 0, incx, 0, y, 0, incy, 0, batch_count);
    }

} // namespace

/* ============================================================================================ */

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocblas_scopy_batched(rocblas_handle     handle,
                                     rocblas_int        n,
                                     const float* const x[],
                                     rocblas_int        incx,
                                     float* const       y[],
                                     rocblas_int        incy,
                                     rocblas_int        batch_count)
try
{
    constexpr rocblas_int NB = 256;
    return rocblas_copy_batched_impl<NB>(handle, n, x, incx, y, incy, batch_count);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_dcopy_batched(rocblas_handle      handle,
                                     rocblas_int         n,
                                     const double* const x[],
                                     rocblas_int         incx,
                                     double* const       y[],
                                     rocblas_int         incy,
                                     rocblas_int         batch_count)
try
{
    constexpr rocblas_int NB = 256;
    return rocblas_copy_batched_impl<NB>(handle, n, x, incx, y, incy, batch_count);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_hcopy_batched(rocblas_handle            handle,
                                     rocblas_int               n,
                                     const rocblas_half* const x[],
                                     rocblas_int               incx,
                                     rocblas_half* const       y[],
                                     rocblas_int               incy,
                                     rocblas_int               batch_count)
try
{
    constexpr rocblas_int NB = 256;
    return rocblas_copy_batched_impl<NB>(handle, n, x, incx, y, incy, batch_count);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_ccopy_batched(rocblas_handle                     handle,
                                     rocblas_int                        n,
                                     const rocblas_float_complex* const x[],
                                     rocblas_int                        incx,
                                     rocblas_float_complex* const       y[],
                                     rocblas_int                        incy,
                                     rocblas_int                        batch_count)
try
{
    constexpr rocblas_int NB = 256;
    return rocblas_copy_batched_impl<NB>(handle, n, x, incx, y, incy, batch_count);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_zcopy_batched(rocblas_handle                      handle,
                                     rocblas_int                         n,
                                     const rocblas_double_complex* const x[],
                                     rocblas_int                         incx,
                                     rocblas_double_complex* const       y[],
                                     rocblas_int                         incy,
                                     rocblas_int                         batch_count)
try
{
    constexpr rocblas_int NB = 256;
    return rocblas_copy_batched_impl<NB>(handle, n, x, incx, y, incy, batch_count);
}
catch(...)
{
    return exception_to_rocblas_status();
}

} // extern "C"
