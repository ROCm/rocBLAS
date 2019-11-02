/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "rocblas_rotm.hpp"
#include "handle.h"
#include "logging.h"
#include "rocblas.h"
#include "utility.h"

namespace
{
    constexpr int NB = 512;

    template <typename>
    constexpr char rocblas_rotm_name[] = "unknown";
    template <>
    constexpr char rocblas_rotm_name<float>[] = "rocblas_srotm";
    template <>
    constexpr char rocblas_rotm_name<double>[] = "rocblas_drotm";

    template <class T>
    rocblas_status rocblas_rotm_impl(rocblas_handle handle,
                                     rocblas_int    n,
                                     T*             x,
                                     rocblas_int    incx,
                                     T*             y,
                                     rocblas_int    incy,
                                     const T*       param)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        auto layer_mode = handle->layer_mode;
        if(layer_mode & rocblas_layer_mode_log_trace)
            log_trace(handle, rocblas_rotm_name<T>, n, x, incx, y, incy, param);
        if(layer_mode & rocblas_layer_mode_log_bench)
            log_bench(handle,
                      "./rocblas-bench -f rotm -r",
                      rocblas_precision_string<T>,
                      "-n",
                      n,
                      "--incx",
                      incx,
                      "--incy",
                      incy);
        if(layer_mode & rocblas_layer_mode_log_profile)
            log_profile(handle, rocblas_rotm_name<T>, "N", n, "incx", incx, "incy", incy);

        if(!x || !y || !param)
            return rocblas_status_invalid_pointer;

        RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);

        return rocblas_rotm_template<NB, false>(
            handle, n, x, 0, incx, 0, y, 0, incy, 0, param, 0, 0, 1);
    }

} // namespace

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

ROCBLAS_EXPORT rocblas_status rocblas_srotm(rocblas_handle handle,
                                            rocblas_int    n,
                                            float*         x,
                                            rocblas_int    incx,
                                            float*         y,
                                            rocblas_int    incy,
                                            const float*   param)
{
    return rocblas_rotm_impl(handle, n, x, incx, y, incy, param);
}

ROCBLAS_EXPORT rocblas_status rocblas_drotm(rocblas_handle handle,
                                            rocblas_int    n,
                                            double*        x,
                                            rocblas_int    incx,
                                            double*        y,
                                            rocblas_int    incy,
                                            const double*  param)
{
    return rocblas_rotm_impl(handle, n, x, incx, y, incy, param);
}

} // extern "C"
