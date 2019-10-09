/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "handle.h"
#include "logging.h"
#include "rocblas.h"
#include "rocblas_rotmg.hpp"
#include "utility.h"

namespace
{
    template <typename>
    constexpr char rocblas_rotmg_name[] = "unknown";
    template <>
    constexpr char rocblas_rotmg_name<float>[] = "rocblas_srotmg_strided_batched";
    template <>
    constexpr char rocblas_rotmg_name<double>[] = "rocblas_drotmg_strided_batched";

    template <class T>
    rocblas_status rocblas_rotmg_strided_batched_impl(rocblas_handle handle,
                                                      T*             d1,
                                                      rocblas_stride stride_d1,
                                                      T*             d2,
                                                      rocblas_stride stride_d2,
                                                      T*             x1,
                                                      rocblas_stride stride_x1,
                                                      const T*       y1,
                                                      rocblas_stride stride_y1,
                                                      T*             param,
                                                      rocblas_stride stride_param,
                                                      rocblas_int    batch_count)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        auto layer_mode = handle->layer_mode;
        if(layer_mode & rocblas_layer_mode_log_trace)
            log_trace(handle,
                      rocblas_rotmg_name<T>,
                      d1,
                      stride_d1,
                      d2,
                      stride_d2,
                      x1,
                      stride_x1,
                      y1,
                      stride_y1,
                      param,
                      batch_count);
        if(layer_mode & rocblas_layer_mode_log_bench)
            log_bench(handle,
                      "./rocblas-bench -f rotmg_strided_batched -r",
                      rocblas_precision_string<T>,
                      "--batch",
                      batch_count,
                      "--stride_a",
                      stride_d1,
                      "--stride_b",
                      stride_d2,
                      "--stride_x",
                      stride_x1,
                      "--stride_y",
                      stride_y1);
        if(layer_mode & rocblas_layer_mode_log_profile)
            log_profile(handle, rocblas_rotmg_name<T>, "batch", batch_count);

        if(!d1 || !d2 || !x1 || !y1 || !param)
            return rocblas_status_invalid_pointer;
        if(batch_count < 0)
            return rocblas_status_invalid_size;

        RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);

        return rocblas_rotmg_template(handle,
                                      d1,
                                      0,
                                      stride_d1,
                                      d2,
                                      0,
                                      stride_d2,
                                      x1,
                                      0,
                                      stride_x1,
                                      y1,
                                      0,
                                      stride_y1,
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

ROCBLAS_EXPORT rocblas_status rocblas_srotmg_strided_batched(rocblas_handle handle,
                                                             float*         d1,
                                                             rocblas_stride stride_d1,
                                                             float*         d2,
                                                             rocblas_stride stride_d2,
                                                             float*         x1,
                                                             rocblas_stride stride_x1,
                                                             const float*   y1,
                                                             rocblas_stride stride_y1,
                                                             float*         param,
                                                             rocblas_stride stride_param,
                                                             rocblas_int    batch_count)
{
    return rocblas_rotmg_strided_batched_impl(handle,
                                              d1,
                                              stride_d1,
                                              d2,
                                              stride_d2,
                                              x1,
                                              stride_x1,
                                              y1,
                                              stride_y1,
                                              param,
                                              stride_param,
                                              batch_count);
}

ROCBLAS_EXPORT rocblas_status rocblas_drotmg_strided_batched(rocblas_handle handle,
                                                             double*        d1,
                                                             rocblas_stride stride_d1,
                                                             double*        d2,
                                                             rocblas_stride stride_d2,
                                                             double*        x1,
                                                             rocblas_stride stride_x1,
                                                             const double*  y1,
                                                             rocblas_stride stride_y1,
                                                             double*        param,
                                                             rocblas_stride stride_param,
                                                             rocblas_int    batch_count)
{
    return rocblas_rotmg_strided_batched_impl(handle,
                                              d1,
                                              stride_d1,
                                              d2,
                                              stride_d2,
                                              x1,
                                              stride_x1,
                                              y1,
                                              stride_y1,
                                              param,
                                              stride_param,
                                              batch_count);
}

} // extern "C"
