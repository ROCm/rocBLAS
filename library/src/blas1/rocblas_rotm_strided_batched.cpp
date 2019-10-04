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
    constexpr char rocblas_rotm_name<float>[] = "rocblas_srotm_strided_batched";
    template <>
    constexpr char rocblas_rotm_name<double>[] = "rocblas_drotm_strided_batched";

    template <class T>
    rocblas_status rocblas_rotm_strided_batched_impl(rocblas_handle handle,
                                                     rocblas_int    n,
                                                     T*             x,
                                                     rocblas_int    incx,
                                                     rocblas_stride stride_x,
                                                     T*             y,
                                                     rocblas_int    incy,
                                                     rocblas_stride stride_y,
                                                     const T*       param,
                                                     rocblas_stride stride_param,
                                                     rocblas_int    batch_count)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        auto layer_mode = handle->layer_mode;
        if(layer_mode & rocblas_layer_mode_log_trace)
            log_trace(handle,
                      rocblas_rotm_name<T>,
                      n,
                      x,
                      incx,
                      stride_x,
                      y,
                      incy,
                      stride_y,
                      param,
                      batch_count);
        if(layer_mode & rocblas_layer_mode_log_bench)
            log_bench(handle,
                      "./rocblas-bench -f rotm_strided_batched -r",
                      rocblas_precision_string<T>,
                      "-n",
                      n,
                      "--incx",
                      incx,
                      "--stride_x",
                      stride_x,
                      "--incy",
                      incy,
                      "--stride_y",
                      stride_y,
                      "--batch",
                      batch_count);
        if(layer_mode & rocblas_layer_mode_log_profile)
            log_profile(handle,
                        rocblas_rotm_name<T>,
                        "N",
                        n,
                        "incx",
                        incx,
                        "stride_x",
                        stride_x,
                        "incy",
                        incy,
                        "stride_y",
                        stride_y,
                        "batch",
                        batch_count);

        if(!x || !y || !param)
            return rocblas_status_invalid_pointer;
        if(batch_count < 0)
            return rocblas_status_invalid_size;

        RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);

        return rocblas_rotm_template<NB, true>(handle,
                                               n,
                                               x,
                                               0,
                                               incx,
                                               stride_x,
                                               y,
                                               0,
                                               incy,
                                               stride_y,
                                               param,
                                               0,
                                               stride_param,
                                               batch_count);
    }

} // namespace

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

ROCBLAS_EXPORT rocblas_status rocblas_srotm_strided_batched(rocblas_handle handle,
                                                            rocblas_int    n,
                                                            float*         x,
                                                            rocblas_int    incx,
                                                            rocblas_stride stride_x,
                                                            float*         y,
                                                            rocblas_int    incy,
                                                            rocblas_stride stride_y,
                                                            const float*   param,
                                                            rocblas_stride stride_param,
                                                            rocblas_int    batch_count)
{
    return rocblas_rotm_strided_batched_impl(
        handle, n, x, incx, stride_x, y, incy, stride_y, param, stride_param, batch_count);
}

ROCBLAS_EXPORT rocblas_status rocblas_drotm_strided_batched(rocblas_handle handle,
                                                            rocblas_int    n,
                                                            double*        x,
                                                            rocblas_int    incx,
                                                            rocblas_stride stride_x,
                                                            double*        y,
                                                            rocblas_int    incy,
                                                            rocblas_stride stride_y,
                                                            const double*  param,
                                                            rocblas_stride stride_param,
                                                            rocblas_int    batch_count)
{
    return rocblas_rotm_strided_batched_impl(
        handle, n, x, incx, stride_x, y, incy, stride_y, param, stride_param, batch_count);
}

} // extern "C"
