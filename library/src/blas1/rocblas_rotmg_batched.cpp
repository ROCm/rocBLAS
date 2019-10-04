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
    constexpr char rocblas_rotmg_name<float>[] = "rocblas_srotmg_batched";
    template <>
    constexpr char rocblas_rotmg_name<double>[] = "rocblas_drotmg_batched";

    template <class T>
    rocblas_status rocblas_rotmg_batched_impl(rocblas_handle handle,
                                              T* const       d1[],
                                              T* const       d2[],
                                              T* const       x1[],
                                              const T* const y1[],
                                              T* const       param[],
                                              rocblas_int    batch_count)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        auto layer_mode = handle->layer_mode;
        if(layer_mode & rocblas_layer_mode_log_trace)
            log_trace(handle, rocblas_rotmg_name<T>, d1, d2, x1, y1, param, batch_count);
        if(layer_mode & rocblas_layer_mode_log_bench)
            log_bench(handle,
                      "./rocblas-bench -f rotmg_batched -r",
                      rocblas_precision_string<T>,
                      "--batch",
                      batch_count);
        if(layer_mode & rocblas_layer_mode_log_profile)
            log_profile(handle, rocblas_rotmg_name<T>, "batch", batch_count);

        if(!d1 || !d2 || !x1 || !y1 || !param)
            return rocblas_status_invalid_pointer;
        if(batch_count < 0)
            return rocblas_status_invalid_size;

        RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);

        return rocblas_rotmg_template(
            handle, d1, 0, 0, d2, 0, 0, x1, 0, 0, y1, 0, 0, param, 0, 0, batch_count);
    }

} // namespace

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

ROCBLAS_EXPORT rocblas_status rocblas_srotmg_batched(rocblas_handle     handle,
                                                     float* const       d1[],
                                                     float* const       d2[],
                                                     float* const       x1[],
                                                     const float* const y1[],
                                                     float* const       param[],
                                                     rocblas_int        batch_count)
{
    return rocblas_rotmg_batched_impl(handle, d1, d2, x1, y1, param, batch_count);
}

ROCBLAS_EXPORT rocblas_status rocblas_drotmg_batched(rocblas_handle      handle,
                                                     double* const       d1[],
                                                     double* const       d2[],
                                                     double* const       x1[],
                                                     const double* const y1[],
                                                     double* const       param[],
                                                     rocblas_int         batch_count)
{
    return rocblas_rotmg_batched_impl(handle, d1, d2, x1, y1, param, batch_count);
}

} // extern "C"
