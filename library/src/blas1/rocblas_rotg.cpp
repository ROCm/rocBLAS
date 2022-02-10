/* ************************************************************************
 * Copyright 2016-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "rocblas_rotg.hpp"
#include "handle.hpp"
#include "logging.hpp"
#include "rocblas.h"
#include "utility.hpp"

namespace
{
    template <typename>
    constexpr char rocblas_rotg_name[] = "unknown";
    template <>
    constexpr char rocblas_rotg_name<float>[] = "rocblas_srotg";
    template <>
    constexpr char rocblas_rotg_name<double>[] = "rocblas_drotg";
    template <>
    constexpr char rocblas_rotg_name<rocblas_float_complex>[] = "rocblas_crotg";
    template <>
    constexpr char rocblas_rotg_name<rocblas_double_complex>[] = "rocblas_zrotg";

    template <class T, class U>
    rocblas_status rocblas_rotg_impl(rocblas_handle handle, T* a, T* b, U* c, T* s)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);

        auto layer_mode     = handle->layer_mode;
        auto check_numerics = handle->check_numerics;
        if(layer_mode & rocblas_layer_mode_log_trace)
            log_trace(handle, rocblas_rotg_name<T>, a, b, c, s);
        if(layer_mode & rocblas_layer_mode_log_bench)
            log_bench(handle,
                      "./rocblas-bench -f rotg --a_type",
                      rocblas_precision_string<T>,
                      "--b_type",
                      rocblas_precision_string<U>);
        if(layer_mode & rocblas_layer_mode_log_profile)
            log_profile(handle, rocblas_rotg_name<T>);

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
                                                       1,
                                                       check_numerics,
                                                       is_input);
            if(rotg_check_numerics_status != rocblas_status_success)
                return rotg_check_numerics_status;
        }

        rocblas_status status
            = rocblas_rotg_template(handle, a, 0, 0, b, 0, 0, c, 0, 0, s, 0, 0, 1);
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
                                                       1,
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

rocblas_status rocblas_srotg(rocblas_handle handle, float* a, float* b, float* c, float* s)
try
{
    return rocblas_rotg_impl(handle, a, b, c, s);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_drotg(rocblas_handle handle, double* a, double* b, double* c, double* s)
try
{
    return rocblas_rotg_impl(handle, a, b, c, s);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_crotg(rocblas_handle         handle,
                             rocblas_float_complex* a,
                             rocblas_float_complex* b,
                             float*                 c,
                             rocblas_float_complex* s)
try
{
    return rocblas_rotg_impl(handle, a, b, c, s);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_zrotg(rocblas_handle          handle,
                             rocblas_double_complex* a,
                             rocblas_double_complex* b,
                             double*                 c,
                             rocblas_double_complex* s)
try
{
    return rocblas_rotg_impl(handle, a, b, c, s);
}
catch(...)
{
    return exception_to_rocblas_status();
}

} // extern "C"
