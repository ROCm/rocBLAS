/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "handle.h"
#include "logging.h"
#include "rocblas.h"
#include "rocblas_rotg.hpp"
#include "utility.h"

namespace
{
    template <typename>
    constexpr char rocblas_rotg_name[] = "unknown";
    template <>
    constexpr char rocblas_rotg_name<float>[] = "rocblas_srotg_batched";
    template <>
    constexpr char rocblas_rotg_name<double>[] = "rocblas_drotg_batched";
    template <>
    constexpr char rocblas_rotg_name<rocblas_float_complex>[] = "rocblas_crotg_batched";
    template <>
    constexpr char rocblas_rotg_name<rocblas_double_complex>[] = "rocblas_zrotg_batched";

    template <class T, class U>
    rocblas_status rocblas_rotg_batched_impl(rocblas_handle handle,
                                             T* const       a[],
                                             T* const       b[],
                                             U* const       c[],
                                             T* const       s[],
                                             rocblas_int    batch_count)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        auto layer_mode = handle->layer_mode;
        if(layer_mode & rocblas_layer_mode_log_trace)
            log_trace(handle, rocblas_rotg_name<T>, a, b, c, s, batch_count);
        if(layer_mode & rocblas_layer_mode_log_bench)
            log_bench(handle,
                      "./rocblas-bench -f rotg_batched --a_type",
                      rocblas_precision_string<T>,
                      "--b_type",
                      rocblas_precision_string<U>,
                      "--batch",
                      batch_count);
        if(layer_mode & rocblas_layer_mode_log_profile)
            log_profile(handle, rocblas_rotg_name<T>, "batch", batch_count);

        if(!a || !b || !c || !s)
            return rocblas_status_invalid_pointer;
        if(batch_count < 0)
            return rocblas_status_invalid_size;

        RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);

        return rocblas_rotg_template(handle, a, 0, 0, b, 0, 0, c, 0, 0, s, 0, 0, batch_count);
    }

} // namespace

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocblas_srotg_batched(rocblas_handle handle,
                                     float* const   a[],
                                     float* const   b[],
                                     float* const   c[],
                                     float* const   s[],
                                     rocblas_int    batch_count)
{
    return rocblas_rotg_batched_impl(handle, a, b, c, s, batch_count);
}

rocblas_status rocblas_drotg_batched(rocblas_handle handle,
                                     double* const  a[],
                                     double* const  b[],
                                     double* const  c[],
                                     double* const  s[],
                                     rocblas_int    batch_count)
{
    return rocblas_rotg_batched_impl(handle, a, b, c, s, batch_count);
}

rocblas_status rocblas_crotg_batched(rocblas_handle               handle,
                                     rocblas_float_complex* const a[],
                                     rocblas_float_complex* const b[],
                                     float* const                 c[],
                                     rocblas_float_complex* const s[],
                                     rocblas_int                  batch_count)
{
    return rocblas_rotg_batched_impl(handle, a, b, c, s, batch_count);
}

rocblas_status rocblas_zrotg_batched(rocblas_handle                handle,
                                     rocblas_double_complex* const a[],
                                     rocblas_double_complex* const b[],
                                     double* const                 c[],
                                     rocblas_double_complex* const s[],
                                     rocblas_int                   batch_count)
{
    return rocblas_rotg_batched_impl(handle, a, b, c, s, batch_count);
}

} // extern "C"
