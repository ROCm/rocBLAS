/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "handle.h"
#include "logging.h"
#include "rocblas.h"
#include "rocblas_rotm.hpp"
#include "utility.h"

namespace
{
    constexpr int NB = 512;

    template <typename>
    constexpr char rocblas_rotm_name[] = "unknown";
    template <>
    constexpr char rocblas_rotm_name<float>[] = "rocblas_srotm_batched";
    template <>
    constexpr char rocblas_rotm_name<double>[] = "rocblas_drotm_batched";

    template <class T>
    rocblas_status rocblas_rotm_batched_impl(rocblas_handle handle,
                                             rocblas_int    n,
                                             T* const       x[],
                                             rocblas_int    incx,
                                             T* const       y[],
                                             rocblas_int    incy,
                                             const T* const param[],
                                             rocblas_int    batch_count)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        auto layer_mode = handle->layer_mode;
        if(layer_mode & rocblas_layer_mode_log_trace)
            log_trace(handle, rocblas_rotm_name<T>, n, x, incx, y, incy, param, batch_count);
        if(layer_mode & rocblas_layer_mode_log_bench)
            log_bench(handle,
                      "./rocblas-bench -f rotm_batched -r",
                      rocblas_precision_string<T>,
                      "-n",
                      n,
                      "--incx",
                      incx,
                      "--incy",
                      incy,
                      "--batch",
                      batch_count);
        if(layer_mode & rocblas_layer_mode_log_profile)
            log_profile(handle,
                        rocblas_rotm_name<T>,
                        "N",
                        n,
                        "incx",
                        incx,
                        "incy",
                        incy,
                        "batch",
                        batch_count);

        if(!x || !y || !param)
            return rocblas_status_invalid_pointer;
        if(batch_count < 0)
            return rocblas_status_invalid_size;

        RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);

        return rocblas_rotm_template<NB, true>(
            handle, n, x, 0, incx, 0, y, 0, incy, 0, param, 0, 0, batch_count);
    }

} // namespace

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

ROCBLAS_EXPORT rocblas_status rocblas_srotm_batched(rocblas_handle     handle,
                                                    rocblas_int        n,
                                                    float* const       x[],
                                                    rocblas_int        incx,
                                                    float* const       y[],
                                                    rocblas_int        incy,
                                                    const float* const param[],
                                                    rocblas_int        batch_count)
{
    return rocblas_rotm_batched_impl(handle, n, x, incx, y, incy, param, batch_count);
}

ROCBLAS_EXPORT rocblas_status rocblas_drotm_batched(rocblas_handle      handle,
                                                    rocblas_int         n,
                                                    double* const       x[],
                                                    rocblas_int         incx,
                                                    double* const       y[],
                                                    rocblas_int         incy,
                                                    const double* const param[],
                                                    rocblas_int         batch_count)
{
    return rocblas_rotm_batched_impl(handle, n, x, incx, y, incy, param, batch_count);
}

} // extern "C"
