/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "handle.hpp"
#include "logging.hpp"
#include "rocblas/rocblas.h"
#include "rocblas_rotg.hpp"
#include "utility.hpp"

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

        RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);

        auto layer_mode     = handle->layer_mode;
        auto check_numerics = handle->check_numerics;
        if(layer_mode & rocblas_layer_mode_log_trace)
            log_trace(handle, rocblas_rotg_name<T>, a, b, c, s, batch_count);
        if(layer_mode & rocblas_layer_mode_log_bench)
            log_bench(handle,
                      "./rocblas-bench -f rotg_batched --a_type",
                      rocblas_precision_string<T>,
                      "--b_type",
                      rocblas_precision_string<U>,
                      "--batch_count",
                      batch_count);
        if(layer_mode & rocblas_layer_mode_log_profile)
            log_profile(handle, rocblas_rotg_name<T>, "batch_count", batch_count);

        if(batch_count <= 0)
            return rocblas_status_success;

        if(!a || !b || !c || !s)
            return rocblas_status_invalid_pointer;

        if(check_numerics)
        {
            bool           is_input = true;
            rocblas_status rotg_check_numerics_status
                = rocblas_rotg_check_numerics_template(rocblas_rotg_name<T>,
                                                       handle,
                                                       1,
                                                       a,
                                                       0,
                                                       0,
                                                       b,
                                                       0,
                                                       0,
                                                       c,
                                                       0,
                                                       0,
                                                       s,
                                                       0,
                                                       0,
                                                       batch_count,
                                                       check_numerics,
                                                       is_input);
            if(rotg_check_numerics_status != rocblas_status_success)
                return rotg_check_numerics_status;
        }

        rocblas_status status
            = rocblas_rotg_template(handle, a, 0, 0, b, 0, 0, c, 0, 0, s, 0, 0, batch_count);
        if(status != rocblas_status_success)
            return status;

        if(check_numerics)
        {
            bool           is_input = false;
            rocblas_status rotg_check_numerics_status
                = rocblas_rotg_check_numerics_template(rocblas_rotg_name<T>,
                                                       handle,
                                                       1,
                                                       a,
                                                       0,
                                                       0,
                                                       b,
                                                       0,
                                                       0,
                                                       c,
                                                       0,
                                                       0,
                                                       s,
                                                       0,
                                                       0,
                                                       batch_count,
                                                       check_numerics,
                                                       is_input);
            if(rotg_check_numerics_status != rocblas_status_success)
                return rotg_check_numerics_status;
        }
        return status;
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
try
{
    return rocblas_rotg_batched_impl(handle, a, b, c, s, batch_count);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_drotg_batched(rocblas_handle handle,
                                     double* const  a[],
                                     double* const  b[],
                                     double* const  c[],
                                     double* const  s[],
                                     rocblas_int    batch_count)
try
{
    return rocblas_rotg_batched_impl(handle, a, b, c, s, batch_count);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_crotg_batched(rocblas_handle               handle,
                                     rocblas_float_complex* const a[],
                                     rocblas_float_complex* const b[],
                                     float* const                 c[],
                                     rocblas_float_complex* const s[],
                                     rocblas_int                  batch_count)
try
{
    return rocblas_rotg_batched_impl(handle, a, b, c, s, batch_count);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_zrotg_batched(rocblas_handle                handle,
                                     rocblas_double_complex* const a[],
                                     rocblas_double_complex* const b[],
                                     double* const                 c[],
                                     rocblas_double_complex* const s[],
                                     rocblas_int                   batch_count)
try
{
    return rocblas_rotg_batched_impl(handle, a, b, c, s, batch_count);
}
catch(...)
{
    return exception_to_rocblas_status();
}

} // extern "C"
